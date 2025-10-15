use num_traits::Num;
use crate::Tensor;

impl<T: Copy + Num> Tensor<T> {
    /// 軸の順序を入れ替える
    /// 2次元ならば転置
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

        for (i, &axis) in axes.iter().enumerate() {
            assert!(axis < self.rank(), "Axis index out of bounds");
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }
        
        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }
}
