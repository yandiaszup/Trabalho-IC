//
//  BiasNeuron.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import Foundation

class BiasNeuron: Neuron {
    init(weights: [Double]) {
        super.init(weights: weights, activationFunction: { _ in return 0.0 }, derivativeActivationFunction: { _ in return 0.0 })
    }
    override func output(inputs: [Double]) -> Double {
        return 1.0
    }
}
