
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'seperate':
        #assert(opt.train_mode == 'seperate')
        from .seperate_model import SeperateModel
        model = SeperateModel()
    elif opt.model == 'cycle':
        #assert(opt.train_mode == 'joint')
        from .cycle_model import CycleModel
        model = CycleModel()
    elif opt.model == 'test':
        #assert(opt.test_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
