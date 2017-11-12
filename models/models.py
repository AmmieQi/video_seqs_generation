
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'seperate':
        #assert(opt.train_mode == 'seperate')
        from .seperate_model import SeperateModel
        model = SeperateModel()
    elif opt.model == 'pred':
        from .cycle_model import CycleModel
        model = CycleModel()
    elif opt.model == 'multi':
        from .multi_model import MultiModel
        model = MultiModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
