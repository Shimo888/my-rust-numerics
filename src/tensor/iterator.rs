use num_traits::Num;
use crate::Tensor;

impl<T> Tensor<T>
where T: Copy + Num{
    /// テンソルのイテレータを取得
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// for (indices, value) in tensor.iter() {
    ///     if indices == vec![0, 0] {
    ///         assert_eq!(value, 1);
    ///     }else if indices == vec![0, 1] {
    ///         assert_eq!(value, 2);
    ///     } else if indices == vec![1, 0] {
    ///         assert_eq!(value, 3);
    ///     } else if indices == vec![1, 1] {
    ///         assert_eq!(value, 4);
    ///     }
    /// }
    /// ```
    pub fn iter(&self) -> TensorIter<T> {
        TensorIter::new(self)
    }
}

/// テンソルのイテレータ構造体
pub struct TensorIter<'a, T>
where T: Copy + Num{
    tensor: &'a Tensor<T>,
    indices: Vec<usize>,
    finished: bool,
}

impl<'a,T> TensorIter<'a,T>
where T: Copy + Num{
    /// テンソルイテレータの新規生成
    pub(super) fn new(tensor : &'a Tensor<T>) -> Self{
        let indices = vec![0; tensor.rank()];
        Self{
            tensor: &tensor,
            indices,
            finished: false,
        }
    }
}

impl<'a, T> Iterator for TensorIter<'a, T>
where T: Copy + Num{
    type Item = (Vec<usize>, T);
    
    /// イテレータの次の要素を取得
    fn next(&mut self) -> Option<Self::Item> {
        
        // イテレーションが終了している場合はNoneを返す
        if self.finished {
            return None;
        }
        
        // 最後の次元のindexからインクリメントしていく
        for dim in (0..self.tensor.rank()).rev() {
            
            // 繰り上がりがないときはこの次元でインクリメントして終了
            if self.indices[dim] + 1 < self.tensor.shape[dim] {
                self.indices[dim] += 1;
                break;
            }
            
            // dimが0のときは終了
            if dim == 0 {
                self.finished = true;
                return None;
            }
            
            // 繰り上がりがあるときはこの次元を0にして次の上位次元へ
            self.indices[dim] = 0;
        }
        
        Some((self.indices.clone(), self.tensor[&self.indices]))
    }
}