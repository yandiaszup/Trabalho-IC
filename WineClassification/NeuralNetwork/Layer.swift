//
//  Layer.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

class Layer {
    let previousLayer: Layer?
    var neurons: [Neuron]
    var outputCache: [Double]
    var hasBias: Bool = false
    
    init(previousLayer: Layer? = nil, neurons: [Neuron] = [Neuron]()) {
        self.previousLayer = previousLayer
        self.neurons = neurons
        self.outputCache = Array<Double>(repeating: 0.0, count: neurons.count)
    }
    
    init(previousLayer: Layer? = nil, numNeurons: Int, activationFunction: @escaping (Double) -> Double, derivativeActivationFunction: @escaping (Double)-> Double, hasBias: Bool = false) {
        self.previousLayer = previousLayer
        self.neurons = Array<Neuron>()
        self.hasBias = hasBias
        for _ in 0..<numNeurons {
            self.neurons.append(Neuron(weights: randomWeights(number: previousLayer?.neurons.count ?? 0), activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction))
        }
        if hasBias {
            self.neurons.append(BiasNeuron(weights: randomWeights(number: previousLayer?.neurons.count ?? 0)))
        }
        self.outputCache = Array<Double>(repeating: 0.0, count: neurons.count)
    }
    
    func outputs(inputs: [Double]) -> [Double] {
        if previousLayer == nil { // camada de entrada
            outputCache = hasBias ? inputs + [1.0] : inputs
        } else { // Camada de saida ou escondida
            outputCache = neurons.map { $0.output(inputs: inputs) }
        }
        return outputCache
    }
    
    // Calcula Gradiente para neuronios da camada de saida
    func calculateDeltasForOutputLayer(expected: [Double]) {
        for n in 0..<neurons.count {
            let error = (expected[n] - neurons[n].activationFunction(neurons[n].inputCache))
            let gradient = neurons[n].derivativeActivationFunction(neurons[n].inputCache) * error
            neurons[n].gradient = gradient
        }
    }
    
    // Calcula Gradientes para neuronios das camadas escondidas
    func calculateDeltasForHiddenLayer(nextLayer: Layer) {
        for (index, neuron) in neurons.enumerated() {
            let nextWeights = nextLayer.neurons.map { $0.weights[index] }
            let nextGradient = nextLayer.neurons.map { $0.gradient }
            let sumOfWeightsXGradients = dotProduct(nextWeights, nextGradient)
            let gradient = neuron.derivativeActivationFunction(neuron.inputCache) * sumOfWeightsXGradients
            neuron.gradient = gradient
        }
    }
}
