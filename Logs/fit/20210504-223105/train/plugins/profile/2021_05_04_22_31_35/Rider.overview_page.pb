?'	?????a@?????a@!?????a@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?????a@?ɩ?a"(@1??????_@A??X?v??I}$%=-@*?n??
m@b??"?Z?@2?
VIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map;Qis+@!???/Q@)??*øc+@1CQIQ@:Preprocessing2?
LIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map?v1?t?1@!c˄i!V@)Ƨ ? @1FJD?)@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map=)?ڐ3@!???JeX@)(*?T??1KDb?;""@:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::MapD?bԵv.@!B|G??R@)C??f??1?{?@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTake::FlatMap[0]::TFRecord|{נ/???!??a.A???)|{נ/???1??a.A???:Advanced file read2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4????Kq??!/EB?Z??)????Kq??1/EB?Z??:Preprocessing2?
qIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImplN??????!#>?????)]Ot]????1C??%$??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2?Ljh???!?x???)?Ljh???1?x???:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch???ϝ`??!???????)???ϝ`??1???????:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality ??G????!dU??]???)gc%?YI??1j $????:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatchS?'?ݚ?!?f~????)S?'?ݚ?1?f~????:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTake::FlatMapoe??2???!?y?i????){???ɚ?1?V?A???:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::ForeverRepeat?V?/?'??!??/????)?Z?7?q??1?V??>??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FiniteTakeB@??
??!}ю2i???) (??{??1??J?ף?:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::ForeverRepeat::MapAndBatch::Map::Map::Map::Map::Prefetch::MemoryCacheC??up???!g??Z????)&4I,)w?1!Z? ˝??:Preprocessing2F
Iterator::Modelk??=]??!??@?pN??)?;?y?9{?1ܰ[@4???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??|y???!yX5+J/??)Qf?L2rv?1?O׹????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??hB?s&@Q????1V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ɩ?a"(@?ɩ?a"(@!?ɩ?a"(@      ??!       "	??????_@??????_@!??????_@*      ??!       2	??X?v????X?v??!??X?v??:	}$%=-@}$%=-@!}$%=-@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??hB?s&@y????1V@?"e
9gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_,??og??!_,??og??0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}"?????!?^3ߙ??0"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???q???!??????0"e
=gradient_tape/model/up_sampling2d_3/resize/ResizeBilinearGradResizeBilinearGradf?f?Ec??!U???J??"4
model/conv2d_16/Conv2DConv2D ?̎?>??!?Z?????0"c
8gradient_tape/model/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput?/??I??!?@?W????0"e
9gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW&{;???!XE??a{??0"c
8gradient_tape/model/conv2d_18/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Ѿ??w??!??Y???0"-
IteratorGetNext/_4_Recv?qH???!?-]????"c
8gradient_tape/model/conv2d_14/Conv2D/Conv2DBackpropInputConv2DBackpropInput?A??@???!??̅????0Q      Y@Y??????5@a^?G??S@q?}FJ2??y/??U?YY?"?	
both?Your program is POTENTIALLY input-bound because 8.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 