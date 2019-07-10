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
    var lastDelta = [Double]()
    var gradient: Double = 0.0
    
    init(weights: [Double], activationFunction: @escaping (Double) -> Double, derivativeActivationFunction: @escaping (Double) -> Double) {
        self.weights = weights
        self.activationFunction = activationFunction
        self.derivativeActivationFunction = derivativeActivationFunction
        for _ in 0...weights.count {
            lastDelta.append(0.0)
        }
    }
    
    // Calcula saida do neuronio baseado nas entradas
    func output(inputs: [Double]) -> Double {
        inputCache = dotProduct(inputs, weights)
        return activationFunction(inputCache)
    }
}
