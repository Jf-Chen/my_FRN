        elif level == 'class':
            psi_output = self.psi(support_embeddings) 
            # support_embeddings torch.Size([25, 25, 640])
            # psi_output [25, 25, 1280]
            
            
            rho_input = torch.cat([psi_output, support_embeddings], dim=2)
            #rho_input torch.Size([25, 25, 1920])
            
            
            rho_input = rho_input.view(self.train_way, self.train_shot,self.resolution, -1)
            # rho_input torch.Size([5, 20, 25, 480])
            
            rho_input = torch.sum(rho_input, dim=1) # [5, 25, 480]
            
            import pdb
            pdb.set_trace()
            
            rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 
            
            pdb.set_trace()
            
            return rho_output
        # level = balance 不要了
        
        support_embeddings [25, 25, 640]
        
        rho_input 应该是[5, 25, 640]为什么对不上
        
而在FRN的meta_test中
inp: torch.Size([105, 3, 84, 84]) way: 5 shot 5 query_shot 16
        
        
具体的原因是glofa的MLP在train阶段就固定形状了，比如train_shot=20,
而当tm.evaluate(model)时，query shot=16,但实际上FRN的seff.f_class仍然是按照20构建的，因此在后面不匹配
        