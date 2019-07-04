////
////  WineClassification.swift
////  WineClassification
////
////  Created by Yan Dias on 26/06/19.
////  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
////
//
import Foundation


class WineClassificator {
    
    let dataParser = DataParser()
    
    var network: Network = Network(layerStructure: [11,22,10], learningRate: 0.1, momentum: 0.9, hasBias: true)
    
    var network2: Network = Network(layerStructure: [11,22,10], learningRate: 0.1, momentum: 0.0, hasBias: true)
    
    // for training
    var wineParameters: [[Double]] = [[Double]]()
    var wineClassifications: [[Double]] = [[Double]]()

    
    var outputs = [Double]()
    
    func start(maxError: Double) {
        self.setInitialWeightsForSecondNetwork()
        
        guard let wineData = dataParser.parseData() else {
            return
        }
        
        splitWineData(data: wineData)
        wineClassifications = createTargetOutputVectors(outputs: outputs)
        
        trainNeuralNetwork(maxError: maxError)
        trainNeuralNetworkWithouMomentum(maxError: maxError)
        
    }
    
    func trainNeuralNetworkWithouMomentum(maxError: Double) {
        var error = Double.infinity
        var firstError = 0.0
        var numberOfCicles = 0
        
        let start = DispatchTime.now()
        while (error > maxError) {
            error = network2.train(inputs: wineParameters, expecteds: wineClassifications)
            if numberOfCicles == 0 {
                firstError = error
            }
            numberOfCicles += 1
//            print(error)
        }
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        
        print("\n\nBackpropagation without momentum benchmark")
        print("Number of cicles \(numberOfCicles)")
        print("Number of epochs \(numberOfCicles * wineParameters.count)")
        print("First EQM \(firstError)")
        print("Final EQM \(error)")
        print("Training time: \(Double(nanoTime)/1000000000) seconds")
        
    }
    
    func setInitialWeightsForSecondNetwork() {
        
        for i in 0...network2.layers[1].neurons.count-1 {
            network2.layers[1].neurons[i].weights = network.layers[1].neurons[i].weights
        }
        
        for i in 0...network2.layers[2].neurons.count-1 {
            network2.layers[2].neurons[i].weights = network.layers[2].neurons[i].weights
        }
    }
    
    func trainNeuralNetwork(maxError: Double) {
        var error = Double.infinity
        var firstError = 0.0
        var numberOfCicles = 0
        
        let start = DispatchTime.now()
        while (error > maxError) {
            error = network.train(inputs: wineParameters, expecteds: wineClassifications)
            if numberOfCicles == 0 {
                firstError = error
            }
            numberOfCicles += 1
//            print(error)
        }
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        
        print("\n\nBackpropagation with momentum benchmark")
        print("Number of cicles \(numberOfCicles)")
        print("Number of epochs \(numberOfCicles * wineParameters.count)")
        print("First EQM \(firstError)")
        print("Final EQM \(error)")
        print("Training time: \(Double(nanoTime)/1000000000) seconds")
        
    }
    
    
    
    fileprivate func normalizeInputVector(input: [Double]) -> [Double] {
        var normalizedInput = [Double]()
        let minMaxList = dataParser.minMax
        for i in 0...10 {
            let value = input[i]
            let normalizedValue = dataParser.normalizedValue(value: value, maxValue: minMaxList[i].1, minValue: minMaxList[i].0)
            normalizedInput.append(normalizedValue)
        }
        return normalizedInput
    }
    
    
    fileprivate func splitWineData(data: [[Double]]) {
        for x in data {
            var atributes = x
            let output = atributes.remove(at: 11)
            wineParameters.append(atributes)
            outputs.append(output)
        }
    }
    
    fileprivate func createTargetOutputVectors(outputs: [Double]) -> [[Double]] {
        var targetOutputVectors = [[Double]]()
        for output in outputs {
            switch output {
            case 0.0/9.0:
                targetOutputVectors.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 1.0/9.0:
                targetOutputVectors.append([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 2.0/9.0:
                targetOutputVectors.append([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 3.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 4.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 5.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0])
                break
            case 6.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
                break
            case 7.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
                break
            case 8.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
                break
            case 9.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
                break
            default:
                break
            }
        }
        return targetOutputVectors
    }
}
