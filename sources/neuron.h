#pragma once
#include <vector>
#include <stdlib.h>


class Neuron;
typedef std::vector<Neuron> Layer;




struct Connection {
	double deltaWeight, weight;
};




class Neuron {
public:
	Neuron				(unsigned numOutputs, unsigned myIndex);
	void feedForward		(const Layer &prevLayer);
	void calcOutputGradients	(double targetVals);
	void calcHiddenGradients	(const Layer &nextLayer);
	void updateInputWeights		(Layer &prevLayer);


	void setOutputVal(double val) {
		outputVal = val;
	}


	double getOutputVal() const {
		return outputVal;
	}

private:
	unsigned m_myIndex;
	static double randomWeight() {
		return rand() / double(RAND_MAX);
	}


	static double activationFunction					(double x);
	static double activationFunctionDerivative				(double x);
	double sumDOW								(const Layer &nextLayer) const;


	static double eta;
	static double alpha;
	double outputVal;
	double gradient;
	std::vector<Connection> outputWeights;


	char padding[16];
};
