//
//  Functions.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//


import Accelerate
import Foundation

// MARK: Randomization & Statistical Helpers

// gera valores aleatorios de -1.0 a 1.0
func randomWeights(number: Int) -> [Double] {
    return (0..<number).map{ _ in Math.randomFractional() * 2 - 1 }
}

// gera valores do tipo Double aleatorios entre 0.0 e o limite dado
func randomNums(number: Int, limit: Double) -> [Double] {
    return (0..<number).map{ _ in Math.randomTo(limit: limit) }
}

// embaralha os elementos de um array
extension Sequence where Iterator.Element : Comparable {
    var shuffled: [Self.Iterator.Element] {
        return sorted { _, _ in arc4random() % 2 == 0 }
    }
}

// MARK: Funcoes de ativacao e suas derivadas

func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

func derivativeSigmoid(_ x: Double) -> Double {
    let sigRes = sigmoid(x)
    return sigRes * (1 - sigRes)
}

func relu(_ x: Double) -> Double {
    return max(x, 0)
}

func derivateRelu(_ x: Double) -> Double {
    return x >= 0 ? 1 : 0
}

// MARK: SIMD Accelerated Math

// retorna o somatorio da multiplicacao dos valores de xs e ys, assumindo que possuem o mesmo tamanho
func dotProduct(_ xs: [Double], _ ys: [Double]) -> Double {
    var answer: Double = 0.0
    vDSP_dotprD(xs, 1, ys, 1, &answer, vDSP_Length(xs.count))
    return answer
}

// subtrai os valores de dois vetores e retorna outro vetor
public func sub(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](y)
    catlas_daxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

// multiplica os valores de dois vetores e retorna outro vetor
public func mul(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vDSP_vmulD(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

// retorna a soma dos valores do vetor
public func sum(x: [Double]) -> Double {
    var result: Double = 0.0
    vDSP_sveD(x, 1, &result, vDSP_Length(x.count))
    
    return result
}

// MARK: Random Number Generation

// this struct & the randomFractional() function
// based on http://stackoverflow.com/a/35919911/281461
struct Math {
    private static var seeded = false
    
    static func randomFractional() -> Double {
        
        if !Math.seeded {
            let time = Int(NSDate().timeIntervalSinceReferenceDate)
            srand48(time)
            Math.seeded = true
        }
        
        return drand48()
    }
    
    // addition, just multiplies random number by *limit*
    static func randomTo(limit: Double) -> Double {
        
        if !Math.seeded {
            let time = Int(NSDate().timeIntervalSinceReferenceDate)
            srand48(time)
            Math.seeded = true
        }
        
        return drand48() * limit
    }
}
