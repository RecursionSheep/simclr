### Loss 实现

在 `run.py` 中增加一个输入参数表示使用的 loss function。

```python
flags.DEFINE_string(
    'loss_func', 'NT-Xent',
    'Loss function.')
```

在 `objective.py` 中实现 NT-Logistic 和 Margin Triplet 两种 loss function。论文中提到，NT-Logistic 和 Margin Triplet 两种 loss 可以通过实现 semi-hard negative mining 来提升性能，即计算 loss 时，取小于正样本中最接近的负样本反向传播梯度。具体实现时，使用 `tf.greater_equal` 函数比较正负样本的 loss 大小得到 mask，将大于正样本的过滤掉，取最大值即可。

```python
if FLAGS.loss_func != 'NT-Xent':
    logits_positive = tf.diag_part(logits_ab)
    temp_positive = tf.tile(tf.expand_dims(logits_positive, -1), [1, logits_a.shape[1]])
    masks_a = tf.cast(tf.greater_equal(logits_a, temp_positive - 1e-5), tf.float32)
    masks_b = tf.cast(tf.greater_equal(logits_b, temp_positive - 1e-5), tf.float32)
    logits_a = logits_a - masks_a * LARGE_NUM
    logits_b = logits_b - masks_b * LARGE_NUM
    logits_negative_a = tf.reduce_max(logits_a, axis = 1)
    logits_negative_b = tf.reduce_max(logits_b, axis = 1)
    if FLAGS.loss_func == 'NT-Logistic':
        loss_a = tf.reduce_mean(tf.log(1 + tf.exp(-logits_positive)) + tf.log(1 + tf.exp(logits_negative_a)))
        loss_b = tf.reduce_mean(tf.log(1 + tf.exp(-logits_positive)) + tf.log(1 + tf.exp(logits_negative_b)))
        tf.losses.add_loss(loss_a + loss_b)
        return loss_a + loss_b, logits_ab, labels
    else:
        loss_a = tf.reduce_mean(tf.maximum(logits_negative_a - logits_positive + MARGIN, 0))
        loss_b = tf.reduce_mean(tf.maximum(logits_negative_b - logits_positive + MARGIN, 0))
        tf.losses.add_loss(loss_a + loss_b)
        return loss_a + loss_b, logits_ab, labels
```

### 实验结果

测试结果如下：

| Loss function | NT-Xent | NT-Logistic | Margin Triplet |
| Top-1 accuracy | 90.21% | 79.54% | 78.92% |
| Top-5 accuracy | 99.89% | 98.84% | 98.75% |

NT-Logistic 和 Margin Triplet 两种 loss 的结果都明显不如 NT-Xent（设定的 batch size 为默认的512，我估计增大 batch size 还能提升训练性能，但是大 batch size 对单 GPU 训练来说训练时间太长了，所以没有尝试），这与论文中的结果一致。因为 NT-Xent 是将所有正负样本一起计算 cross entropy，反传梯度时考虑了所有负样本的信息。而另外两种 loss 使用 semi-hard negative mining 也只是取一个最接近的样本计算 loss。


