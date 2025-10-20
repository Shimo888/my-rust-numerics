use std::sync::Arc;
use num_traits::Num;
use crate::{Tensor, TensorError};

/// テンソルの形状操作に関する実装
impl<T: Copy + Num> Tensor<T> {
    /// 軸の順序を入れ替える 
    /// 2次元ならば転置 
    /// 実装的にはshapeとstridesを入れ替えるだけで、dataはそのまま(ゼロコピー) 
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// let transposed = tensor.permute_axes(&[1, 0]);
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// assert_eq!(transposed, Tensor::new(vec![1, 4, 2, 5, 3, 6], vec![3, 2]).unwrap());
    /// ```
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.rank(), "Axes length must match tensor rank");
        let mut new_shape = vec![0; self.rank()];
        let mut new_strides = vec![0; self.rank()];

        let mut seen = vec![false; self.rank()];
        for (i, &axis) in axes.iter().enumerate() {
            assert!(axis < self.rank(), "Axis index out of bounds");
            assert!(!seen[axis], "Duplicate axis in permutation");
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
            seen[axis] = true;
        }
        
        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }
    
    /// テンソルが連続したメモリレイアウトかどうかを判定
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert!(tensor.is_contiguous());
    /// let transposed = tensor.permute_axes(&[1, 0]);
    /// assert!(!transposed.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        
        for i in (0..self.rank()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
    
    /// テンソルを連続したメモリレイアウトに変換 
    /// 非連続な場合、dataとstridesを再構築するため、データのコピーが発生する 
    /// Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// let transposed = tensor.permute_axes(&[1, 0]);
    /// let mut contiguous = transposed.clone();
    /// contiguous.convert_contiguous();
    /// assert!(contiguous.is_contiguous());
    /// ```
    pub fn convert_contiguous(&mut self) {
        if self.is_contiguous() {
            return;
        }
        
        let mut new_data = vec![T::zero(); self.data.len()];
        let mut new_strides = Self::calc_contiguous_strides(&self.shape);
        
        // 全要素を新しいデータ配列にコピーして新しいストライドを適用
        for (indices, val) in self.iter() {
            
            let mut flat_index = 0;
            for (i, &index) in indices.iter().enumerate() {
                flat_index += index * new_strides[i];
            }
            
            new_data[flat_index] = val;
        }
        
        self.data = Arc::new(new_data);
        self.strides = new_strides;
    }
    
    /// テンソルの形状を変更(要素数は不変) 
    /// メモリアウトが連続している場合高速に動作する 
    /// メモリアウトが非連続な場合、連続化するためデータのコピーが発生する 
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// let reshaped = tensor.reshape(&[3, 2]).unwrap();
    /// assert_eq!(reshaped.shape(), &[3, 2]);
    /// assert_eq!(reshaped, Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap());
    /// let transposed = tensor.permute_axes(&[1, 0]);
    /// let reshaped_transposed = transposed.reshape(&[3, 2]).unwrap();
    /// assert_eq!(reshaped_transposed.shape(), &[3, 2]);
    /// assert_eq!(reshaped_transposed, Tensor::new(vec![1, 4, 2, 5, 3, 6], vec![3, 2]).unwrap());
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        // 要素数のチェック
        let total_element = self.shape.iter().product::<usize>();
        let new_total_element = new_shape.iter().product::<usize>();
        if total_element != new_total_element {
            return Err(TensorError::ShapeMismatch);
        }
        
        if self.is_contiguous() {
            // 連続のときはゼロコピーでshapeとstridesを変更するだけ
            let new_strides = Self::calc_contiguous_strides(new_shape);
            Ok(Self {
                data: self.data.clone(),
                shape: new_shape.to_vec(),
                strides: new_strides,
            })
        } else{
            let mut contiguous_tensor = self.clone();
            contiguous_tensor.convert_contiguous();
            contiguous_tensor.reshape(new_shape)
        }
    }
}
