---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-2-e39c7da753d1> in <module>()
    132 #-----------end------------
    133 
--> 134 tm.evaluate(model)
    135 
    136 # 没训练完，5-way 5-shot,train_query_shot 5

10 frames
/content/my_FRN/trainers/trainer.py in evaluate(self, model)
    252                                         transform_type=args.test_transform_type,
    253                                         query_shot=args.test_query_shot,
--> 254                                         trial=10000)
    255 
    256                 logger.info('%d-way-%d-shot acc: %.2f\t%.2f'%(args.test_way,shot,mean,interval))

/content/my_FRN/trainers/eval.py in meta_test(data_path, model, way, shot, pre, transform_type, query_shot, trial, return_list)
     37         # pdb.set_trace()
     38 
---> 39         max_index = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)
     40 
     41         acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

/content/my_FRN/models/FRN.py in meta_test(self, inp, way, shot, query_shot)
    191                                         way=way,
    192                                         shot=shot,
--> 193                                         query_shot=query_shot)
    194 
    195         _,max_index = torch.max(neg_l2_dist,1)

/content/my_FRN/models/FRN.py in get_neg_l2_dist(self, inp, way, shot, query_shot, return_support)
    135         query_embeddings = feature_map[train_way*train_shot:] # [75,25,640]
    136         mask_task=self.f_task(support_embeddings,level='task')
--> 137         mask_class=self.f_class(support_embeddings, level='class')
    138 
    139 

/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

/content/my_FRN/models/set_function.py in forward(self, support_embeddings, level)
     53             rho_input = torch.sum(rho_input, dim=1)
     54 
---> 55             rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6
     56             return rho_output
     57         # level = balance 不要了

/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

/usr/local/lib/python3.7/dist-packages/torch/nn/modules/container.py in forward(self, input)
    137     def forward(self, input):
    138         for module in self:
--> 139             input = module(input)
    140         return input
    141 

/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

/usr/local/lib/python3.7/dist-packages/torch/nn/modules/linear.py in forward(self, input)
     94 
     95     def forward(self, input: Tensor) -> Tensor:
---> 96         return F.linear(input, self.weight, self.bias)
     97 
     98     def extra_repr(self) -> str:

/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py in linear(input, weight, bias)
   1845     if has_torch_function_variadic(input, weight):
   1846         return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
-> 1847     return torch._C._nn.linear(input, weight, bias)
   1848 
   1849 

RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`