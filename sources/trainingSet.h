#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class TrainingSet
{
public:
	bool isEOF() {
		return trainingDataFile.eof();
	}

	TrainingSet			(const std::string filename);
	void getTopology		(std::vector<unsigned> &topology);
	unsigned getNextInputs		(std::vector<double> &inputVals);
	unsigned getTargetOutputs	(std::vector<double> &targetOutputVals);

private:
	std::ifstream trainingDataFile;


	char padding[56];
};
