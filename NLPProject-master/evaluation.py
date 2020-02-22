#  Created by Takshak Desai and Mithun Balakrishna on 9/22/19.
#  Copyright 2019 Takshak Desai. All rights reserved.

import sys
import os
import math

GOLD = 0
PREDICTED = 1
label_set = [1, 2, 3, 4, 5]

def argument_parser():
	if len(sys.argv) != 3:
		sys.exit("Usage: python evaluation.py gold_file predicted_file")
	if not os.path.isfile(sys.argv[1]) and not os.path.isfile(sys.argv[1]):
		sys.exit("Input to script should be a file!")
	return sys.argv[1], sys.argv[2]

def get_scores(true_tags, predicted_tags, prec = 3):
	classification_report = dict()
	classification_report['micro'] = {'recall':0.0, 'precision':0.0, 'fscore':0.0}
	for label in label_set:
		classification_report[label] = {'recall':0.0, 'precision':0.0, 'fscore':0.0}
		tp, fp, fn = 0, 0, 0
		for key in true_tags.keys():
			gold_tag = true_tags[key]
			predicted_tag = predicted_tags[key]
			if gold_tag == predicted_tag:
				print (predicted_tag, label)
				if predicted_tag == label:
					tp +=1
			else:
				if predicted_tag == label:
					fp +=1
				else:
					fn +=1
		try:
			recall = float(tp)/(tp+fn)
		except ZeroDivisionError:
			recall = 0.0
		try:
			precision = float(tp)/(tp+fp)
		except ZeroDivisionError:
			precision = 0.0
		try:
			fscore = 2*precision*recall/(precision+recall)
		except ZeroDivisionError:
			fscore = 0.0
		classification_report[label]['recall'] = round(recall, prec)
		classification_report[label]['precision'] = round(precision, prec)
		classification_report[label]['fscore'] = round(fscore, prec)
		classification_report['micro']['recall'] += recall
		classification_report['micro']['precision'] += precision
		classification_report['micro']['fscore'] += fscore

	for key in classification_report['micro'].keys():
		classification_report['micro'][key] /= len(label_set)
		classification_report['micro'][key] = round(classification_report['micro'][key], prec)
	return classification_report

def get_correlation(gold_tags, predicted_tags):
	sigma_x, sigma_y, sigma_xy, sigma_x2, sigma_y2 = 0.0, 0.0, 0.0, 0.0, 0.0
	for key in gold_tags.keys():
		sigma_x += gold_tags[key]
		sigma_y += predicted_tags[key]
		sigma_x2 += gold_tags[key]**2
		sigma_y2 += predicted_tags[key]**2
		sigma_xy += gold_tags[key]*predicted_tags[key]
	n = len(gold_tags)
	r = n*sigma_xy - sigma_x*sigma_y
	r /= math.sqrt((n*sigma_x2 - sigma_x**2)*(n*sigma_y2 - sigma_y**2))
	return r

def pretty_print(report):
	print("Class\tP\tR\tF1\n")
	for label in report.keys():
		if label == 'micro':
			print 
		print (str(label) + "\t" + str(report[label]['precision']) + "\t" + str(report[label]['recall']) + "\t" + str(report[label]['fscore'])) 


def file_reader(file_path, mode = GOLD):
	sent_to_tag = dict()
	try:
		reader = open(file_path, 'r', encoding="utf8")
		data = reader.readlines()
		if mode == GOLD:
			data = data[1:]
		for line in data:
			data = line.split("\t")
			sent_to_tag[data[0]] = int(data[-1].replace("\n", ""))
		reader.close()
		return sent_to_tag
	except IOError:
		sys.exit("File not found at location " + file_path)

if __name__ == "__main__":
	gold_file, predicted_file = argument_parser()
	#gold_file = "./data/dev-set.txt"
	#predicted_file = "./sample_predictions.txt"
	gold_tags = file_reader(gold_file)
	predicted_tags = file_reader(predicted_file, mode = GOLD)
	try:
		assert (gold_tags.keys() == predicted_tags.keys())
	except AssertionError:
		sys.exit("Gold and predicted file do not contain same number of predictions!")
	classification_report = get_scores(gold_tags, predicted_tags)
	pretty_print(classification_report)

	correlation_score = get_correlation(gold_tags, predicted_tags)
	print ("\nPearson correlation coefficient: ", correlation_score)
