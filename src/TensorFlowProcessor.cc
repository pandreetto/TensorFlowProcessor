#include "TensorFlowProcessor.h"
#include "marlin/VerbosityLevels.h"

#include "tensorflow/lite/interpreter_builder.h"

using tflite::InterpreterBuilder;

TensorFlowProcessor aTensorFlowProcessor;

TensorFlowProcessor::TensorFlowProcessor() :
    Processor("TensorFlowProcessor"),
    model(),
    resolver(),
    interpreter()
{
    _description = "Simple test processor for tensorflow-lite";

    registerProcessorParameter("TF-model-file", 
                               "Filename of the tensorflow model",
                               _tfmodelfile,
                               string(""));
}

void TensorFlowProcessor::init()
{
    streamlog_out(MESSAGE) << "Initializing TensorFlowProcessor" << std::endl;
    model = FlatBufferModel::BuildFromFile(_tfmodelfile.c_str());
    streamlog_out(MESSAGE) << "Loaded model " << _tfmodelfile.c_str() << std::endl;
    InterpreterBuilder(*model, resolver)(&interpreter);
}

void TensorFlowProcessor::processRunHeader(LCRunHeader* run)
{}

void TensorFlowProcessor::processEvent(LCEvent * evt)
{}

void TensorFlowProcessor::check(LCEvent *evt)
{}

void TensorFlowProcessor::end()
{}

