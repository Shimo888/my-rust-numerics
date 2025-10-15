use std::ops::{Index, IndexMut};
use std::sync::Arc;
use num_traits::Num;
use crate::Tensor;

impl<T: Copy + Num> Tensor<T>{ 
    /// インデックスアクセス
    /// # Examples
    /// [[1, 2, 3],
    /// [4, 5, 6]]
    /// tensor.get(&[1, 2]) -> Some(&6)
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(tensor.get(&[1, 2]), Some(&6)); 
    /// assert_eq!(tensor.get(&[2, 2]), None); // インデックスが範囲外
    /// assert_eq!(tensor.get(&[1, 2, 1]), None); // インデックスの次元数が不正
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T>{
        if indices.len() != self.rank() {
            return None;
        }
    
        let flat_index = self.compute_flat_index(indices);
    
        if flat_index.is_none(){
            return None;
        }
    
        if self.data.len() <= flat_index.unwrap() {
            return None;
        }
    
        self.data.get(flat_index.unwrap())
    }
    
    
    /// ミュータブルなインデックスアクセス
    /// # Examples
    /// [[1, 2, 3],
    /// [4, 5, 6]]
    /// tensor.get_mut(&[1, 2]) -> Some(&mut 6)
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// if let Some(value) = tensor.get_mut(&[1, 2]) {
    ///   *value = 10;
    /// }
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T>{
        if indices.len() != self.rank() {
            return None;
        }
    
        let flat_index = self.compute_flat_index(indices)?;
    
        if self.data.len() <= flat_index {
            return None;
        }

        // 書き込み時コピーをすることで、同時編集を避ける
        // dataのRCカウントが1より大きい場合、データをクローンしてミュータブルにする
        // dataのRCカウントが1の場合、そのままミュータブルにする
        let data_mut = Arc::make_mut(&mut self.data);
        data_mut.get_mut(flat_index)  
    }
    
    /// shapeが[6,5,7]の([3,4,2])のインデックスを計算する場合
    /// (5 * 7) * 3 + (7) * 4 + (1) * 2
    /// 1, 7, 5*7 がstride
    fn compute_flat_index(&self, indices: &[usize]) -> Option<usize>{
        if indices.len() != self.rank(){
            return None;
        }
        
        let flat_index = indices.iter()
            .zip(self.strides.iter())
            .map(|(&idx, stride)| idx * stride)
            .sum();
        Some(flat_index)
    }
}

/// IndexトレイトとIndexMutトレイトの実装
impl <T:Copy + Num> Index<&[usize]> for Tensor<T>{
    type Output = T;
    /// インデックスアクセス
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// assert_eq!(tensor[&[1, 2]], 6);
    /// ```
    /// panic if out of bounds
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// // This will panic
    /// // let _ = tensor[&[2, 2]];
    /// ```
    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get(indices).expect("Tensor index out of bounds")
    }
}

/// IndexMutトレイトの実装
impl <T:Copy + Num> IndexMut<&[usize]> for Tensor<T> {
    /// ミュータブルなインデックスアクセス
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// tensor[&[1, 2]] = 10;
    /// assert_eq!(tensor[&[1, 2]], 10);
    /// ```
    /// panic if out of bounds
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let mut tensor = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    /// // This will panic
    /// // tensor[&[2, 2]] = 10;
    /// ```
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        self.get_mut(indices).expect("Tensor index out of bounds")
    }
}