//
//  Neuron.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//


class Neuron {
    var weights: [Double]
    var activationFunction: (Double) -> Double
    var derivativeActivationFunction: (Double) -> Double
    var inputCache: Double = 0.0
    var delta: Double = 0.0
    var learningRate: Double
    var momentum: Double
    
    init(weights: [Double], activationFunction: @escaping (Double) -> Double, derivativeActivationFunction: @escaping (Double) -> Double, learningRate: Double = 0.25, momentum: Double = 0.20) {
        self.weights = weights
        self.activationFunction = activationFunction
        self.derivativeActivationFunction = derivativeActivationFunction
        self.learningRate = learningRate
        self.momentum = momentum
    }
    
    // Calcula saida do neuronio baseado nas entradas
    func output(inputs: [Double]) -> Double {
        inputCache = dotProduct(inputs, weights)
        return activationFunction(inputCache)
    }
    
}
