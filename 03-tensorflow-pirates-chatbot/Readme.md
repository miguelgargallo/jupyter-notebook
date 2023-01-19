In case you got a similar error

still stuck:

```
1

#training the model
train = model.fit(x_train,y_train,epochs=200)

Epoch 1/200

2023-01-19 14:07:32.719434: W tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:326] libdevice is required by this HLO module but was not found at ./libdevice.10.bc
2023-01-19 14:07:32.719780: W tensorflow/core/framework/op_kernel.cc:1830] OP_REQUIRES failed at xla_ops.cc:446 : INTERNAL: libdevice not found at ./libdevice.10.bc
2023-01-19 14:07:32.734790: W tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:326] libdevice is required by this HLO module but was not found at ./libdevice.10.bc
2023-01-19 14:07:32.735059: W tensorflow/core/framework/op_kernel.cc:1830] OP_REQUIRES failed at xla_ops.cc:446 : INTERNAL: libdevice not found at ./libdevice.10.bc

---------------------------------------------------------------------------
InternalError                             Traceback (most recent call last)
Cell In[131], line 2
      1 #training the model
----> 2 train = model.fit(x_train,y_train,epochs=200)

File ~/.local/lib/python3.10/site-packages/keras/utils/traceback_utils.py:70, in filter_traceback.<locals>.error_handler(*args, **kwargs)
     67     filtered_tb = _process_traceback_frames(e.__traceback__)
     68     # To get the full stack trace, call:
     69     # `tf.debugging.disable_traceback_filtering()`
---> 70     raise e.with_traceback(filtered_tb) from None
     71 finally:
     72     del filtered_tb

File ~/.local/lib/python3.10/site-packages/tensorflow/python/eager/execute.py:52, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     50 try:
     51   ctx.ensure_initialized()
---> 52   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     53                                       inputs, attrs, num_outputs)
     54 except core._NotOkStatusException as e:
     55   if name is not None:

InternalError: Graph execution error:

Detected at node 'StatefulPartitionedCall_5' defined at (most recent call last):
    File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
      return _run_code(code, main_globals, None,
    File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
      exec(code, run_globals)
    File "/usr/lib/python3.10/site-packages/ipykernel_launcher.py", line 17, in <module>
      app.launch_new_instance()
    File "/usr/lib/python3.10/site-packages/traitlets/config/application.py", line 1041, in launch_instance
      app.start()
    File "/usr/lib/python3.10/site-packages/ipykernel/kernelapp.py", line 724, in start
      self.io_loop.start()
    File "/usr/lib/python3.10/site-packages/tornado/platform/asyncio.py", line 215, in start
      self.asyncio_loop.run_forever()
    File "/usr/lib/python3.10/asyncio/base_events.py", line 603, in run_forever
      self._run_once()
    File "/usr/lib/python3.10/asyncio/base_events.py", line 1906, in _run_once
      handle._run()
    File "/usr/lib/python3.10/asyncio/events.py", line 80, in _run
      self._context.run(self._callback, *self._args)
    File "/usr/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 512, in dispatch_queue
      await self.process_one()
    File "/usr/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 501, in process_one
      await dispatch(*args)
    File "/usr/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 408, in dispatch_shell
      await result
    File "/usr/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 731, in execute_request
      reply_content = await reply_content
    File "/usr/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 417, in do_execute
      res = shell.run_cell(
    File "/usr/lib/python3.10/site-packages/ipykernel/zmqshell.py", line 540, in run_cell
      return super().run_cell(*args, **kwargs)
    File "/usr/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 2945, in run_cell
      result = self._run_cell(
    File "/usr/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3000, in _run_cell
      return runner(coro)
    File "/usr/lib/python3.10/site-packages/IPython/core/async_helpers.py", line 129, in _pseudo_sync_runner
      coro.send(None)
    File "/usr/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3203, in run_cell_async
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    File "/usr/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3382, in run_ast_nodes
      if await self.run_code(code, result, async_=asy):
    File "/usr/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3442, in run_code
      exec(code_obj, self.user_global_ns, self.user_ns)
    File "/tmp/ipykernel_40257/48579934.py", line 2, in <module>
      train = model.fit(x_train,y_train,epochs=200)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/engine/training.py", line 1685, in fit
      tmp_logs = self.train_function(iterator)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/engine/training.py", line 1284, in train_function
      return step_function(self, iterator)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/engine/training.py", line 1268, in step_function
      outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/engine/training.py", line 1249, in run_step
      outputs = model.train_step(data)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/engine/training.py", line 1054, in train_step
      self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 532, in minimize
      self.apply_gradients(grads_and_vars)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 1163, in apply_gradients
      return super().apply_gradients(grads_and_vars, name=name)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 639, in apply_gradients
      iteration = self._internal_apply_gradients(grads_and_vars)
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 1189, in _internal_apply_gradients
      return tf.__internal__.distribute.interim.maybe_merge_call(
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 1239, in _distributed_apply_gradients_fn
      distribution.extended.update(
    File "/home/fengh/.local/lib/python3.10/site-packages/keras/optimizers/optimizer.py", line 1234, in apply_grad_to_update_var
      return self._update_step_xla(grad, var, id(self._var_key(var)))
Node: 'StatefulPartitionedCall_5'
libdevice not found at ./libdevice.10.bc
	 [[{{node StatefulPartitionedCall_5}}]] [Op:__inference_train_function_30698]
```

Here: https://github.com/miguelgargallo/jupyter-notebook/blob/main/03-tensorflow-pirates-chatbot/03-tensorflow-pirates-chatbot.ipynb

But this worked on Google Colab, I want it in local

So here I am

![image](https://user-images.githubusercontent.com/5947268/213451447-cadd1f2e-cd78-4fdf-bc89-e129d8f3c0eb.png)


Do this on console:

```bash
find / -type d -name nvvm 2>/dev/null
```

go to this directory if it's not /usr/lib/nvvm/
create directories: /usr/lib/nvvm/
and place there the file: libdevice.10.bc

then go to your project directory where it fails, and paste the file libdevice.10.bc there too.

This solved my problem <3 good luck!