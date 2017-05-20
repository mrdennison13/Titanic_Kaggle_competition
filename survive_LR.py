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
	'Fare': AS_IS,
	'Embarked': DUMMY,
	'Survived': AS_IS
}

feat_val_tables = {
	'Sex': {'male': 0, 'female': 1}
}
	
def tabular(x, table):
	return table[x]

def dummyvar(x, unique_vals):
	return tuple([ (1 if x==a else 0) for a in unique_vals ])
	
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
		result_unique[f]=list()
	for f in feats_min_max:
		result_min_max[f]=[float('inf'), -float('inf')]
	for row in table:
		for f in feats_unique:
			if row[f] not in result_unique[f]:
				result_unique[f].append(row[f])
		for f in feats_min_max:
			try:
				val = float(row[f])
			except:
				continue
			result_min_max[f][0] = min(val, result_min_max[f][0])
			result_min_max[f][1] = max(val, result_min_max[f][1])
	return result_unique, result_min_max

	
def make_set(table, mode, label_header, keys, unique_vals, min_max_vals):

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

def main():
	train_table, headers_train = read_table('train.csv')
	keys = list(quantisation_rules.keys())
	unique_vals, min_max_vals = get_unique_vals(train_table, keys)	
	training_data, training_labels = make_set(train_table, TRAIN, 'Survived', keys, unique_vals, min_max_vals)
	test_table, headers_test = read_table('test.csv')
	keys.remove('Survived')
	test_data = make_set(test_table, TEST, 'Survived', keys, unique_vals, min_max_vals)
	logreg = lm.LogisticRegression()
	logreg.fit(training_data, training_labels)
	predicted_labels=logreg.predict(test_data)
	
	output_data = test_table
	assert len(output_data)==len(predicted_labels)
	for i in range(len(output_data)):
		output_data[i]['Survived']=predicted_labels[i]
	
	write_table(output_data, headers_train, 'prediction_Kwaadgras_LR.csv') #Score so far: 0.77033

if __name__=="__main__":
	main()