/*
	*This is 'teech' neuronet class
	*Now it's necessarily class to neuronet can complete her task and work right, but it's just milestone of beta version.
*/


#include "trainingSet.h"


TrainingSet::TrainingSet(const std::string filename) {
	trainingDataFile.open(filename.c_str());
}




//set topology of neuronet
void TrainingSet::getTopology(std::vector<unsigned> &topology) {
	std::string line;
	std::string label;


	std::getline(trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEOF() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}




//get input(s) from data.txt
unsigned TrainingSet::getNextInputs(std::vector<double> &inputVals) {
	inputVals.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}




//get outputs from neuronet
unsigned TrainingSet::getTargetOutputs(std::vector<double> &targetOutputVals) {
	targetOutputVals.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}
