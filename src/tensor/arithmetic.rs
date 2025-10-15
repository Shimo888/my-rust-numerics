use std::ops::{Add, Mul, Sub};
use std::sync::Arc;
use num_traits::Num;
use crate::{Tensor, TensorError};

impl<T> PartialEq for Tensor<T>
where T: Copy + Num + PartialEq{
    /// テンソルの等価比較
    fn eq(&self, other: &Self) -> bool {
        if self.shape !=  other.shape {
            return false;
        }
        
        if self.strides == other.strides && Arc::ptr_eq(&self.data, &other.data) {
            return true;
        }
        
        //// 2Dの例(for の入れ子構造だけでは、NDには対応できない)
        //for i in 0..self.shape[0] {
        //    for j in 0..self.shape[1] {
        //        if self.get(&[i, j]) != other.get(&[i, j]) {
        //            return false;
        //        }
        //    }
        //}
        
        return compare_recursive(self, other, &mut vec![]);
        
        fn compare_recursive<T>(tensor1: &Tensor<T>, tensor2: &Tensor<T>, indices: &mut Vec<usize>) -> bool
        where T: Copy + Num + PartialEq{
            if tensor1.rank() != indices.len() {
                return tensor1.get(indices) == tensor2.get(indices)
            }
            
            let dim = indices.len() - 1;
            for i in 0..tensor1.shape[dim] {
                indices.push(i);
                if !compare_recursive(tensor1, tensor2, indices) {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: Copy + Num> Tensor<T>{
    /// テンソルの加算
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let result = tensor1.add(&tensor2).unwrap();
    /// assert_eq!(result, Tensor::new(vec![6, 8, 10, 12], vec![2, 2]).unwrap());
    /// ```
    pub fn add(&self, other: &Self) -> Result<Tensor<T>, TensorError>{
        if self.shape != other.shape{
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a + *b).collect();
        let data = Arc::new(data);
        Ok(Tensor{data, shape: self.shape.clone(), strides: self.strides.clone()})
    }
    
    /// テンソルの減算
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let result = tensor1.sub(&tensor2).unwrap();
    /// assert_eq!(result, Tensor::new(vec![4, 4, 4, 4], vec![2, 2]).unwrap());
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Tensor<T>, TensorError>{
        if self.shape != other.shape{
            return Err(TensorError::ShapeMismatch); 
        }
        let data: Vec<T> = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a - *b).collect();
        let data = Arc::new(data);
        Ok(Tensor{data, shape: self.shape.clone(), strides: self.strides.clone()})
    }
    
    /// アダマール積
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let result = tensor1.hadamard_product(&tensor2).unwrap();
    /// assert_eq!(result, Tensor::new(vec![5, 12, 21, 32], vec![2, 2]).unwrap());  
    /// ```
    pub fn hadamard_product(&self, other: &Self) -> Result<Tensor<T>, TensorError>{
        if self.shape != other.shape{
            return Err(TensorError::ShapeMismatch); 
        }
        let data: Vec<T> = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a * *b).collect();
        let data = Arc::new(data);
        Ok(Tensor{data, shape: self.shape.clone(), strides: self.strides.clone()})
    }
}

/// + 演算子のオーバーロード
// impl Add<Rhs(右側)> for Lhs(左側)
impl<'a,'b, T> Add<&'b Tensor<T>> for &'a Tensor<T> 
where T: Copy + Num{ 
    type Output = Tensor<T>;

    /// テンソルの加算(演算子オーバーロード)
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let result = &tensor1 + &tensor2;
    /// assert_eq!(result, Tensor::new(vec![6, 8, 10, 12], vec![2, 2]).unwrap());
    /// ```
    fn add(self, other: &'b Tensor<T>) -> Self::Output {
        self.add(other).unwrap()
    }
}

/// - 演算子のオーバーロード
impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T> 
where T: Copy + Num{
    type Output = Tensor<T>;

    /// テンソルの減算(演算子オーバーロード)
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let result = &tensor1 - &tensor2;
    /// assert_eq!(result, Tensor::new(vec![4, 4, 4, 4], vec![2, 2]).unwrap());
    /// ```
    fn sub(self, other: &'b Tensor<T>) -> Self::Output {
        self.sub(other).unwrap()
    }
}

impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T> 
where T: Copy + Num{
    type Output = Tensor<T>;

    /// テンソルのアダマール積(演算子オーバーロード)
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    /// let result = &tensor1 * &tensor2;
    /// assert_eq!(result, Tensor::new(vec![5, 12, 21, 32], vec![2, 2]).unwrap());
    /// ```
    fn mul(self, other: &'b Tensor<T>) -> Self::Output {
        self.hadamard_product(other).unwrap()
    }
}