use num_traits::Num;
use crate::Tensor;

impl<T:Copy + Num> Tensor<T> {
    /// 指定した形状と値でテンソルを生成
    /// # Examples
    /// ```
    /// use my_rust_numerics::Tensor;
    /// let tensor = Tensor::full(&[2, 3], 5);
    /// assert_eq!(tensor, Tensor::new(vec![5, 5, 5, 5, 5, 5], vec![2, 3]).unwrap());
    /// ```
    pub fn full(shape: &[usize], value: T) -> Self {
        let size = shape.iter().product();
        let data = vec![value; size];
        let strides = Self::calc_contiguous_strides(shape);
        let data = std::sync::Arc::new(data);
        Self { data, shape: shape.to_vec(), strides }
    }

    /// 指定した形状で0のテンソルを生成
    pub fn zeros(shape:&[usize]) -> Self{
        Self::full(shape, T::zero())
    }
    
    /// 指定した形状で1のテンソルを生成
    pub fn ones(shape:&[usize]) -> Self{
        Self::full(shape, T::one())
    }
}