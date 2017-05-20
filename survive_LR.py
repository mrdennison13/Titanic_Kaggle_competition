import csv
import sklearn
import sklearn.linear_model as lm

AS_IS=0
DUMMY=1
TABLE=2

TRAIN=0
TEST=1

quantisation_rules = {
	'Pclass': AS_IS,
	'Sex': TABLE,
	'Age': AS_IS,
	'SibSp': AS_IS,
	'Parch': AS_IS,
	'Fare': AS_IS,
	'nCabins': AS_IS,
	'Title': DUMMY,
	'Embarked': DUMMY,
	'CabinLetters': DUMMY,
	'Survived': AS_IS
}

feat_val_tables = {
	'Sex': {'male': 0, 'female': 1}
}
	
def tabular(x, table):
	return table[x]

def dummyvar(x, unique_vals):
	if isinstance(x, str) or isinstance(x, int) or isinstance(x, float):
		#return tuple([ (1 if x==a else 0) for a in unique_vals ])
		return dummyvar([x], unique_vals)
	elif isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
		return tuple([ (1 if a in x else 0) for a in unique_vals ])
	
def read_table(filename):
	result = list()
	with open(filename, 'r') as in_f:
		inreader = csv.reader(in_f)
		headers = next(inreader)
		for row in inreader:
			assert len(row)==len(headers)
			toappend = dict()
			for i in range(len(headers)):
				toappend[headers[i]]=row[i].strip()
			result.append(toappend)
	return result, headers

def write_table(data, headers, filename):
	with open(filename, 'w') as out_f:
		outwriter = csv.writer(out_f, delimiter=',', lineterminator='\n')
		outwriter.writerow(headers)
		for row in data:
			outwriter.writerow([ row[h] for h in headers ])

def get_unique_vals(table, keys):
	feats_unique = list()
	feats_min_max = list()
	for key in keys:
		if quantisation_rules[key] in (DUMMY, ):
			feats_unique.append(key)
		if quantisation_rules[key] in (AS_IS, ):
			feats_min_max.append(key)

			result_unique = dict()
	result_min_max = dict()
	for f in feats_unique:
		result_unique[f]=set()
	for f in feats_min_max:
		result_min_max[f]=[float('inf'), -float('inf')]
	for row in table:
		for f in feats_unique:
			if isinstance(row[f], set):
				result_unique[f]=result_unique[f].union(row[f])
			else:
				result_unique[f].add(row[f])
		for f in feats_min_max:
			try:
				val = float(row[f])
			except:
				continue
			result_min_max[f][0] = min(val, result_min_max[f][0])
			result_min_max[f][1] = max(val, result_min_max[f][1])
	return result_unique, result_min_max

def add_derived_features(table):
	for row in table:
	
		#find title (let's see if this works :P)
		name_part = row['Name'].split(',')[-1].strip()
		title = name_part.split(' ')[0].strip()
		row['Title']=title
		
		#find number of cabins & cabin letters
		cabins = row['Cabin'].split(' ')
		n_cabins = len(cabins)
		cabin_letters = set()
		for c in cabins:
			to_add = ''.join([i for i in str(c.strip()) if not i.isdigit()])
			cabin_letters.add(to_add)
		row['nCabins']=1 if n_cabins==0 else n_cabins #it seems reasonable to assume most people have 1 cabin
		row['CabinLetters']=cabin_letters
	
def make_q_headers(keys, unique_vals):
	result = list()
	for key in keys:
		rule = quantisation_rules[key]
		if rule==DUMMY:
			for val in unique_vals[key]:
				result.append(key + '_' + val)
		else:
			result.append(key)
	return tuple(result)
	
def make_t_set(table, mode, label_header, keys, unique_vals, min_max_vals):

	result_data = list()
	result_labels = list()
	
	for row in table:
		result_row = list()
		label_quant = None
		for key in keys:
			rule = quantisation_rules[key]
			if rule==AS_IS:
				if row[key]=='':
					val = [ 0.5 ]
				else:
					val=[ (float(row[key]) - min_max_vals[key][0])/(min_max_vals[key][1] - min_max_vals[key][0]) ]
			elif rule==DUMMY:
				val=dummyvar(row[key], unique_vals[key])
			elif rule==TABLE:
				val=[tabular(row[key], feat_val_tables[key])]
			if key == label_header:
				label_quant = val[0]
			else:
				result_row.extend(val)
		if mode==TRAIN:
			assert label_quant is not None
		result_data.append(tuple(result_row))
		if mode==TRAIN:
			result_labels.append(row[label_header])
	if mode==TEST:
		return result_data
	else:
		return result_data, result_labels

def get_accuracy(model, data, real_labels):
	pred_labels = model.predict(data)
	corr = 0
	assert len(pred_labels)==len(real_labels)
	for i in range(len(pred_labels)):
		if str(pred_labels[i])==str(real_labels[i]):
			corr+=1
	return float(corr) / len(pred_labels)

def main():
	train_table, headers_train = read_table('train.csv')
	add_derived_features(train_table)
	write_table(train_table, headers_train + ['Title', 'nCabins', 'CabinLetters'], 'derived_table.csv')


	keys = list(quantisation_rules.keys())
	unique_vals, min_max_vals = get_unique_vals(train_table, keys)
	
	training_data, training_labels = make_t_set(train_table, TRAIN, 'Survived', keys, unique_vals, min_max_vals)
	
	test_table, headers_test = read_table('test.csv')
	keys.remove('Survived')
	add_derived_features(test_table)
	test_data = make_t_set(test_table, TEST, 'Survived', keys, unique_vals, min_max_vals)
	logreg = lm.LogisticRegression(verbose=1)
	logreg.fit(training_data, training_labels)
	print()

	coef_headers = make_q_headers(keys, unique_vals)
	assert len(coef_headers)==len(logreg.coef_[0])
	for i in range(len(coef_headers)):
		print("{h}: {c}".format(h=coef_headers[i], c=logreg.coef_[0][i]))
	print("Intercept: " + str(logreg.intercept_[0]))
		
	print("\nAccuracy against training set: " + str(get_accuracy(logreg, training_data, training_labels)))
	
	

	predicted_labels=logreg.predict(test_data)	
	output_data = test_table
	assert len(output_data)==len(predicted_labels)
	for i in range(len(output_data)):
		output_data[i]['Survived']=predicted_labels[i]
	write_table(output_data, headers_train + ['Title', 'nCabins', 'CabinLetters'], 'prediction_full_LR.csv')
	write_table(output_data, ['PassengerId', 'Survived'], 'prediction_Kwaadgras_LR.csv') #Score so far: 0.78469

if __name__=="__main__":
	main()