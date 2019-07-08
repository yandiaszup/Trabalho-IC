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

/// Create *number* of random Doubles between -1.0 and 1.0
func randomWeights(number: Int) -> [Double] {
    return (0..<number).map{ _ in Math.randomFractional() * 2 - 1 }
}

/// Create *number* of random Doubles between 0.0 and *limit*
func randomNums(number: Int, limit: Double) -> [Double] {
    return (0..<number).map{ _ in Math.randomTo(limit: limit) }
}

/// primitive shuffle - not fisher yates... not uniform distribution
extension Sequence where Iterator.Element : Comparable {
    var shuffled: [Self.Iterator.Element] {
        return sorted { _, _ in arc4random() % 2 == 0 }
    }
}

// MARK: Funcao de ativacao e sua derivada

func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

func derivativeSigmoid(_ x: Double) -> Double {
    let sigRes = sigmoid(x)
    return sigRes * (1 - sigRes)
}

// MARK: SIMD Accelerated Math

// Based on example from Surge project
// https://github.com/mattt/Surge/blob/master/Source/Arithmetic.swift
/// Find the dot product of two vectors
/// assuming that they are of the same length
/// using SIMD instructions to speed computation
func dotProduct(_ xs: [Double], _ ys: [Double]) -> Double {
    var answer: Double = 0.0
    vDSP_dotprD(xs, 1, ys, 1, &answer, vDSP_Length(xs.count))
    return answer
}

// Based on example from Surge project
// https://github.com/mattt/Surge/blob/master/Source/Arithmetic.swift
/// Subtract one vector from another
/// assuming that they are of the same length
/// using SIMD instructions to speed computation
public func sub(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](y)
    catlas_daxpby(Int32(x.count), 1.0, x, 1, -1, &results, 1)
    
    return results
}

// Another Surge example, see above citation
public func mul(x: [Double], y: [Double]) -> [Double] {
    var results = [Double](repeating: 0.0, count: x.count)
    vDSP_vmulD(x, 1, y, 1, &results, 1, vDSP_Length(x.count))
    
    return results
}

// Another Surge example, see above citation
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
