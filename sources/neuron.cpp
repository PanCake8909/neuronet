/*
	*This is neuron class which uses in net.
	*So, I can say that this is logic kernel.
*/


#include "neuron.h"
#include <math.h>


//settings
double Neuron::eta = 0.15; // net learning rate
double Neuron::alpha = 0.5; // momentum




//construct
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; ++c) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}




//update weight of neuron layer to export data
void Neuron::updateInputWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}




double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}




//caculate count of hidden gradients
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activationFunctionDerivative(outputVal);
}




//calculate count of output layers
void Neuron::calcOutputGradients(double targetVals) {
	double delta = targetVals - outputVal;
	gradient = delta * Neuron::activationFunctionDerivative(outputVal);
}




//neuron activate and complete her task
double Neuron::activationFunction(double x) {
	//output range [-1.0..1.0]
	return tanh(x);
}




double Neuron::activationFunctionDerivative(double x) {
	return 1.0 - x * x;
}




//feed neuron a new data
void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[m_myIndex].weight;
	}

	outputVal = Neuron::activationFunction(sum);
}
