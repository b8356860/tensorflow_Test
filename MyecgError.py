# -*- coding: utf-8 -*-
"""
Self-defined exception
Created on Fri Apr 17 15:08:32 2015
@author: Nale
"""

class MyecgError(Exception):
    """
    Root class for every exception in RPeakDetection
    """
    def __init__(self, *args, **kwargs):
        self.InitializeMessage()
        
    def InitializeMessage(self):
        self.message = self.__class__.__name__ + ': '


class MyecgWarning(UserWarning):
    """
    Root class for every warning in RPeakDetection
    """
    def __init__(self, *args, **kwargs):
        self.InitializeMessage()
        
    def InitializeMessage(self):
        self.message = self.__class__.__name__ + ': '
        
        
#
# Exception and warning in RPeakDetection.py
#
class MyecgCutInError(MyecgError):
    """
    This exception should be thrown out if the cut-in point of signal is
    greater than signal length.
    """
    def __init__(self, **kwargs):
        super(MyecgCutInError, self).__init__()
        self.message += kwargs.get('message')


class MyecgClusterError(MyecgError):
    """
    This exception will be thrown out if there is an index('label', generated
    by scipy.cluster.vq.kmeans2) which does not correspond to currently 
    set 'groupIdx'.
    """
    def __init__(self, **kwargs):
        super(MyecgClusterError, self).__init__()
        self.message += kwargs.get('message')


class MyecgInvalidDataError(MyecgError):
    """
    This exception should be thrown out if there is a problem of ECG data.
    """
    def __init__(self, **kwargs):
        super(MyecgInvalidDataError, self).__init__()
        self.message += kwargs.get('message')


## This is currently replaced by 'MyecgInvalidDataError'.
class MyecgInvalidDataWarning(MyecgWarning):
    """
    This exception should be thrown out if there is a problem of ECG data.
    """
    def __init__(self, **kwargs):
        super(MyecgInvalidDataWarning, self).__init__()
        self.message += 'There might be some problems in this ECG Data.\n'
        self.message += kwargs.get('message')


class MyecgLowRPeakError(MyecgError):
    """
    If R peaks are too small, Pan-Tompkins algorithm might not be able to
    work properly. So this exception will be thrown out, and we can try to use
    lead_01 of ECG data to do analysis again.
    """
    def __init__(self, **kwargs):
        super(MyecgLowRPeakError, self).__init__()
        self.message += kwargs.get('message')


class MyecgLowSignalLevelError(MyecgError):
    """
    This exception will be thrown out if the mean of signal is too small.
    Note: Currently, this exception can only be thrown out from 'case02'
          process in RPeakDetection. 'mean' is actually the mean value of upper
          profile of the signal.
    """
    def __init__(self, **kwargs):
        super(MyecgLowSignalLevelError, self).__init__()
        self.message += kwargs.get('message')


#
# Exception in RPeakanalysis.py:
#
class MyecgValueError(MyecgError):
    """
    This exception should be thrown out if there is an error about the input
    parameter.
    """
    def __init__(self, **kwargs):
        super(MyecgValueError, self).__init__()
        self.message += kwargs.get('message')


class MyecgComputationError(MyecgError):
    """
    This exception should be thrown out if there is any computation error
    occurs.
    """
    def __init__(self, **kwargs):
        super(MyecgComputationError, self).__init__()
        self.message += kwargs.get('message')