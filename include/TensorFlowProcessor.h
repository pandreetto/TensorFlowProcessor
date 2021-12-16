#ifndef TensorFlowProcessor_h
#define TensorFlowProcessor_h 1

#include <string>
#include <vector>

#include "marlin/Processor.h"
#include "lcio.h"
#include "EVENT/LCIO.h"
#include "IMPL/LCCollectionVec.h"

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

using std::string;
using marlin::Processor;
using tflite::FlatBufferModel;
using tflite::ops::builtin::BuiltinOpResolver;

using FlatBufferModelUPtr = std::unique_ptr<FlatBufferModel>;
using InterpreterUPtr = std::unique_ptr<tflite::Interpreter>;

class TensorFlowProcessor : public Processor
{  
public:
  
    virtual Processor*  newProcessor() { return new TensorFlowProcessor ; }

    TensorFlowProcessor();

    /** Called at the begin of the job before anything is read.
    * Use to initialize the processor, e.g. book histograms.
    */
    virtual void init();

    /** Called for every run.
    */
    virtual void processRunHeader( LCRunHeader* run );

    /** Called for every event - the working horse.
    */
    virtual void processEvent( LCEvent * evt ); 

    virtual void check( LCEvent * evt );

    /** Called after data processing for clean up.
    */
    virtual void end();

private:
    string _tfmodelfile;
    FlatBufferModelUPtr model;
    BuiltinOpResolver resolver;
    InterpreterUPtr interpreter;
};

#endif //TensorFlowProcessor_h

