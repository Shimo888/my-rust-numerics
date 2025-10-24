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
        
        for (indices, val) in self.iter(){
            if val != other[&indices] {
                return false;
            }
        }
        true
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
    
    /// テンソルの縮約 
    /// axesで指定した軸を縮約する
    /// # Arguments
    /// - tensor1: 縮約するテンソル1
    /// - tensor2: 縮約するテンソル2
    /// - axes1: tensor1で縮約する軸のインデックス配列
    /// - axes2: tensor2で縮約する軸のインデックス配列
    /// # Returns
    /// - 縮約後のテンソル
    /// # Errors
    /// - TensorError::ShapeMismatch: 縮約する軸のサイズが一致しない場合
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// // 4x2テンソルと2x3テンソルの縮約
    /// let tensor1 = Tensor::new(vec![
    ///    1, 2,
    ///    3, 4,
    ///    5, 6,
    ///    7, 8,], vec![4, 2]).unwrap(); // shape: [4,2]
    /// let tensor2 = Tensor::new(vec![
    ///   1, 2, 3,
    ///   4, 5, 6], vec![2, 3]).unwrap(); // shape: [2,3]
    /// let result = Tensor::tensor_dot(&tensor1, &tensor2, &[1], &[0]).unwrap(); // 縮約軸: tensor1の軸1, tensor2の軸0
    /// assert_eq!(result, Tensor::new(vec![
    ///   9, 12, 15,
    ///   19, 26, 33,
    ///   29, 40, 51,
    ///   39, 54, 69], vec![4, 3]).unwrap()); // shape: [4,3]
    /// ```
    /// 2x1x4テンソルと1x2x4テンソルの縮約 [0,2],[2,0]を縮約
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let data_a: Vec<i64> = (1..=15).collect();
    /// let a = Tensor::ones(&[4,5,6]); 
    /// let b = Tensor::ones(&[5,3,4]);
    /// let result = Tensor::tensor_dot(&a, &b, &[1,0], &[0,2]).unwrap();
    /// println!("{:?}", result.shape()); // [6,3]
    /// assert_eq!(result, Tensor::full(&[6,3], 20))
    /// ```
    /// テンソル積も計算可能である
    /// ``` 
    /// use my_rust_numerics::Tensor;
    /// let tensor1 = Tensor::new(vec![1, 2], vec![2]).unwrap(); // shape: [2]
    /// let tensor2 = Tensor::new(vec![3, 4], vec![2]).unwrap(); // shape: [2]
    /// let result = Tensor::tensor_dot(&tensor1, &tensor2, &[], &[]).unwrap(); // 縮約軸なし
    /// assert_eq!(result, Tensor::new(vec![3, 4, 6, 8], vec![2, 2]).unwrap()); // shape: [2,2]
    /// ```
    pub fn tensor_dot(tensor1 : &Tensor<T>, tensor2 : &Tensor<T>, axes1: &[usize], axes2: &[usize]) -> Result<Tensor<T>, TensorError> {
        // 1. 縮約する軸のチェック
        // 1.1 縮約する軸の数が一致すること
        if axes1.len() != axes2.len() {
            return Err(TensorError::ShapeMismatch);
        }
        
        // 1.2 縮約する軸のサイズが一致すること
        for (&axis1, &axis2) in axes1.iter().zip(axes2.iter()) {
            if tensor1.shape[axis1] != tensor2.shape[axis2] {
                return Err(TensorError::ShapeMismatch);
            }
        }
        
        // 1.3 縮約する軸が有効な範囲内であること
        // 1.4 縮約する軸が重複していないこと
        let mut hashset1 = std::collections::HashSet::new();
        for &axis in axes1 {
            if axis >= tensor1.rank() {
                return Err(TensorError::ShapeMismatch);
            }
            if !hashset1.insert(axis) {
                return Err(TensorError::ShapeMismatch);
            }
        }
        let mut hashset2 = std::collections::HashSet::new();
        for &axis in axes2 {
            if axis >= tensor2.rank() {
                return Err(TensorError::ShapeMismatch);
            }
            if !hashset2.insert(axis) {
                return Err(TensorError::ShapeMismatch);
            }
        }
        
        // 2. 軸の並び替え
        // 自由な軸(free_axes)と縮約する軸(contracted_axes)に分離
        let mut free_axes1 = Vec::new();
        let mut contracted_axes1 = Vec::new();
        for i in 0..tensor1.rank() {
            if axes1.contains(&i) {
                contracted_axes1.push(i);
            } else {
                free_axes1.push(i);
            }
        }
        tensor1.permute_axes(&[free_axes1.as_slice(), contracted_axes1.as_slice()].concat());
        
        let mut free_axes2 = Vec::new();
        let mut contracted_axes2 = Vec::new();
        for i in 0..tensor2.rank() {
            if axes2.contains(&i) {
                contracted_axes2.push(i);
            } else {
                free_axes2.push(i);
            }
        }
        tensor2.permute_axes(&[contracted_axes2.as_slice(), free_axes2.as_slice()].concat());
        
        // 3. 2D行列にReshape
        let reshaped1 =tensor1.reshape(&[
            free_axes1.iter().map(|&axis| tensor1.shape[axis]).product(),
            contracted_axes1.iter().map(|&axis| tensor1.shape[axis]).product(),
        ]).unwrap();
        
        let reshaped2 = tensor2.reshape(&[
            contracted_axes2.iter().map(|&axis| tensor2.shape[axis]).product(),
            free_axes2.iter().map(|&axis| tensor2.shape[axis]).product(),
        ]).unwrap();
        
        let mut new_tensor = Self::internal_mat_mul_2d(&reshaped1, &reshaped2);
        
        // 4. 元の形状にReshape
        let new_shape: Vec<usize> = free_axes1.iter().map(|&axis| tensor1.shape[axis])
            .chain(free_axes2.iter().map(|&axis| tensor2.shape[axis]))
            .collect();
        new_tensor = new_tensor.reshape(&new_shape)?;
        
        Ok(new_tensor)
    }
    
    /// 2D行列式 A[i,j] * B[j,k] = C[i,k]の計算
    /// 内部計算用に使用する
    /// (多次元のテンソルを縮約する際などに、2DにReshapeして計算する)
    /// # Panics
    /// - aまたはbが2Dテンソルでない場合
    /// - aの列数とbの行数が一致しない場合
    /// - aとbがcontiguousでない場合
    fn internal_mat_mul_2d(a_tensor: &Tensor<T>, b_tensor: &Tensor<T>) -> Tensor<T> {
        if a_tensor.rank() != 2 || b_tensor.rank() != 2 {
            panic!("Both tensors must be 2D for internal_mat_mul_2d.");
        }
        
        if a_tensor.shape[1] != b_tensor.shape[0] {
            panic!("Inner dimensions must match for matrix multiplication.");
        }
        
        if !a_tensor.is_contiguous() || !b_tensor.is_contiguous() {
            panic!("Both tensors must be contiguous for internal_mat_mul_2d.");
        }
        
        let mut c_tensor = Tensor::zeros(&[a_tensor.shape[0], b_tensor.shape[1]]);
        let c_data = Arc::make_mut(&mut c_tensor.data);
        
        //// 低速版　(メモリアクセス効率が悪い)
        //for i in 0..a_tensor.shape[0] {
        //    for k in 0..b_tensor.shape[1] {
        //        let mut sum = T::zero();
        //        for j in 0..a_tensor.shape[1] {
        //            // C[i,k] += A[i,j] * B[j,k]
        //            let a_index = i * a_tensor.strides[0] + j * a_tensor.strides[1];
        //            let b_index = j * b_tensor.strides[0] + k * b_tensor.strides[1];
        //            sum = sum + a_tensor.data[a_index] * b_tensor.data[b_index];
        //        }
        //        let result_index = i * c_tensor.strides[0] + k * c_tensor.strides[1];
        //        c_data[result_index] = sum;
        //    }
        //}
        
        // 高速版 (メモリアクセス効率を改善)
        for i in 0..a_tensor.shape[0] {
            for j in 0..a_tensor.shape[1] {
                // A[i,j] 
                let a_index = i * a_tensor.strides[0] + j * a_tensor.strides[1];
                let a_value = a_tensor.data[a_index];
                
                for k in 0..b_tensor.shape[1] {
                    // 1ずつflat_indexが増えるのでメモリのアクセス効率が良い
                    let b_index = j * b_tensor.strides[0] + k * b_tensor.strides[1];
                    let result_index = i * c_tensor.strides[0] + k * c_tensor.strides[1];
                    // C[i,k] += A[i,j] * B[j,k]
                    c_data[result_index] = c_data[result_index] + a_value * b_tensor.data[b_index];
                }
            }
        }
        
        c_tensor
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