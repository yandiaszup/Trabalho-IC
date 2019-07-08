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
    
    init(previousLayer: Layer? = nil, numNeurons: Int, activationFunction: @escaping (Double) -> Double, derivativeActivationFunction: @escaping (Double)-> Double, learningRate: Double, momentum: Double,  hasBias: Bool = false) {
        self.previousLayer = previousLayer
        self.neurons = Array<Neuron>()
        self.hasBias = hasBias
        for _ in 0..<numNeurons {
            self.neurons.append(Neuron(weights: randomWeights(number: previousLayer?.neurons.count ?? 0), activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate, momentum: momentum))
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
    
    // Calcula deltas para camada de saida
    func calculateDeltasForOutputLayer(expected: [Double]) {
        for n in 0..<neurons.count {
            neurons[n].delta = neurons[n].derivativeActivationFunction( neurons[n].inputCache) * (expected[n] - outputCache[n])
        }
    }
    
    // Calcula os deltas para camadas escondidas
    func calculateDeltasForHiddenLayer(nextLayer: Layer) {
        for (index, neuron) in neurons.enumerated() {
            let nextWeights = nextLayer.neurons.map { $0.weights[index] }
            let nextDeltas = nextLayer.neurons.map { $0.delta }
            let sumOfWeightsXDeltas = dotProduct(nextWeights, nextDeltas)
            neuron.delta = neuron.learningRate * neuron.derivativeActivationFunction(neuron.inputCache) * sumOfWeightsXDeltas + (neuron.momentum * neuron.delta)
        }
    }
}
