#pragma once
#include <vector>
#include "neuron.h"


typedef std::vector<Neuron> Layer;

class Net {
public:
	Net						(const std::vector<unsigned> &topology);
	void feedForward				(const std::vector<double> &inputVals);
	void backProp					(const std::vector<double> &targetVals);
	void getResults					(std::vector<double> &resultVals) const;
	
	double getRecentAverageError() const {
		return recentAverageError;
	}

private:
	std::vector<Layer> layers;//layers[layerNumber][NeuronNumber]
	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;

	
	char padding[24];
};
