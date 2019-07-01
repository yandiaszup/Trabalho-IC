//
//  DataParser.swift
//  WineClassification
//
//  Created by Yan Dias on 25/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import Foundation

class DataParser {
    
    var minMax = [(Double,Double)]()
    
    func parseData() -> [[Double]]? {
        let path = Bundle.main.path(forResource: "test", ofType: "txt") ?? ""
        
        do {
            let text2 = try String(contentsOfFile: path, encoding: .utf8)
            let splitedDataSet = text2.split(separator: "\n")
            var dataSetDoubleFormat = [[Double]]()
            for set in splitedDataSet {
                let formatted = set.split(separator: ";")
                var floatSet = [Double]()
                for index in formatted {
                    let value: Double = Double((String(index))) as! Double
                    floatSet.append(value)
                }
                dataSetDoubleFormat.append(floatSet)
            }
            let result = normalize(dataSet: dataSetDoubleFormat)
            return result
        }
        catch {
            return nil
        }
    }
    
    func normalize(dataSet: [[Double]]) -> [[Double]]{
        
        var normalizedDataSet = dataSet
        
        for index in 0...11 {
            var sequence = [Double]()
            for x in dataSet {
                sequence.append(x[index])
            }
            if index==11 {
                minMax.append((0.0,9.0))
            } else {
                minMax.append(takeMinAndMaxValues(sequence: sequence))
            }
        }
        
        for index in 0...11 {
            for x in 0...dataSet.count-1 {
                normalizedDataSet[x][index] = normalizedValue(value: normalizedDataSet[x][index], maxValue: minMax[index].1, minValue: minMax[index].0)
            }
        }
        
        return normalizedDataSet
    }
    
    func normalizedValue(value: Double, maxValue: Double, minValue: Double) -> Double {
        let normalizedValue = (value - minValue)/(maxValue-minValue)
        return normalizedValue
    }
    
    func takeMinAndMaxValues(sequence: [Double]) -> (Double,Double) {
        var maxValue = sequence[0]
        var minValue = sequence[0]
        for value in sequence {
            if value >= maxValue {
                maxValue = value
            }
            if value <= minValue {
                minValue = value
            }
        }
        return (minValue,maxValue)
    }
}



