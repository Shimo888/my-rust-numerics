use std::sync::Arc;
use num_traits::Num;

/// テンソルのエラー列挙型  
#[derive(Debug)]
pub enum TensorError{
    ShapeMismatch,
}

/// テンソル構造体  
/// [3,4]のテンソルの例
/// [[1, 2, 3, 4],
/// [5, 6, 7, 8],
/// [9, 10, 11, 12]]
/// データはフラットなベクタで保持
/// [(0,0)] -> 1
/// [(2,1)] -> 6
/// 変数の説明
/// data: テンソルの要素を格納するフラットなベクタ
/// shape: テンソルの各次元のサイズを格納するベクタ
/// strides: 各次元のストライドを格納するベクタ(転置などをゼロコピーで実現するために使用)
#[derive(Debug, Clone)]
pub struct Tensor<T: Copy>{
    pub(super) data: Arc<Vec<T>>,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

/// テンソルの実装
impl<T: Copy + Num> Tensor<T> {
    /// テンソル構造体の新規生成
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_size = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::ShapeMismatch)
        }
        let strides = Self::calc_strides(&shape);
        let data = Arc::new(data);
        Ok(Self { data, shape, strides })
    }

    ///　テンソルの形状を取得
    /// # Examples
    ///```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// テンソルのランク(次元数)を取得
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// 各次元のストライドを計算
    /// (ex) Shape: [3, 4, 5] -> Strides: [4 * 5, 5, 1]
    pub(super) fn calc_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

#[cfg(test)]
mod tests{
    #[test]
    fn test_calc_strides(){
        let shape = vec![3, 4, 5];
        let strides = crate::tensor::core::Tensor::<i32>::calc_strides(&shape);
        assert_eq!(strides, vec![20, 5, 1]);
    }
}
