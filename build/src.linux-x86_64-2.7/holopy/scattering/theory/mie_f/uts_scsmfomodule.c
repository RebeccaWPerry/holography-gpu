/* File: uts_scsmfomodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * See http://cens.ioc.ee/projects/f2py2e/
 * Generation date: Wed Apr 23 10:55:37 2014
 * $Revision:$
 * $Date:$
 * Do not edit this file directly unless you know what you are doing!!!
 */
#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
/*need_includes0*/

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *uts_scsmfo_error;
static PyObject *uts_scsmfo_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef struct {double r,i;} complex_double;

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (((PyArrayObject *)(capi_ ## var ## _tmp))->nd)
#define old_shape(var,dim) (((PyArrayObject *)(capi_ ## var ## _tmp))->dimensions[dim])
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *v = (int)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *v = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj) || PyUnicode_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = uts_scsmfo_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    *v = PyFloat_AsDouble(obj);
#else
    *v = PyFloat_AS_DOUBLE(obj);
#endif
    return 1;
  }
  tmp = PyNumber_Float(obj);
  if (tmp) {
#ifdef __sgi
    *v = PyFloat_AsDouble(tmp);
#else
    *v = PyFloat_AS_DOUBLE(tmp);
#endif
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj) || PyUnicode_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = uts_scsmfo_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(rotcoef,ROTCOEF)(double*,int*,int*,double*,int*);
extern void F_FUNC(asmfr,ASMFR)(complex_double*,int*,double*,double*,double*,complex_double*);
extern void F_FUNC_US(ms_radial_fields,MS_RADIAL_FIELDS)(complex_double*,int*,double*,double*,double*,complex_double*);
extern void F_FUNC(asm,ASM)(complex_double*,int*,double*,double*,complex_double*);
extern void F_FUNC(sbesjy,SBESJY)(double*,int*,double*,double*,double*,double*,int*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/********************************** rotcoef **********************************/
static char doc_f2py_rout_uts_scsmfo_rotcoef[] = "\
Function signature:\n\
  dc = rotcoef(cbe,kmax,nmax,ndim)\n\
Required arguments:\n"
"  cbe : input float\n"
"  kmax : input int\n"
"  nmax : input int\n"
"  ndim : input int\n"
"Return objects:\n"
"  dc : rank-2 array('d') with bounds (2 * ndim + 1,nmax*(nmax+2)+1+1)";
/* extern void F_FUNC(rotcoef,ROTCOEF)(double*,int*,int*,double*,int*); */
static PyObject *f2py_rout_uts_scsmfo_rotcoef(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,int*,int*,double*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double cbe = 0;
  PyObject *cbe_capi = Py_None;
  int kmax = 0;
  PyObject *kmax_capi = Py_None;
  int nmax = 0;
  PyObject *nmax_capi = Py_None;
  double *dc = NULL;
  npy_intp dc_Dims[2] = {-1, -1};
  const int dc_Rank = 2;
  PyArrayObject *capi_dc_tmp = NULL;
  int capi_dc_intent = 0;
  int ndim = 0;
  PyObject *ndim_capi = Py_None;
  static char *capi_kwlist[] = {"cbe","kmax","nmax","ndim",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOO:uts_scsmfo.rotcoef",\
    capi_kwlist,&cbe_capi,&kmax_capi,&nmax_capi,&ndim_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable nmax */
    f2py_success = int_from_pyobj(&nmax,nmax_capi,"uts_scsmfo.rotcoef() 3rd argument (nmax) can't be converted to int");
  if (f2py_success) {
  /* Processing variable ndim */
    f2py_success = int_from_pyobj(&ndim,ndim_capi,"uts_scsmfo.rotcoef() 4th argument (ndim) can't be converted to int");
  if (f2py_success) {
  /* Processing variable cbe */
    f2py_success = double_from_pyobj(&cbe,cbe_capi,"uts_scsmfo.rotcoef() 1st argument (cbe) can't be converted to double");
  if (f2py_success) {
  /* Processing variable kmax */
    f2py_success = int_from_pyobj(&kmax,kmax_capi,"uts_scsmfo.rotcoef() 2nd argument (kmax) can't be converted to int");
  if (f2py_success) {
  /* Processing variable dc */
  dc_Dims[0]=2 * ndim + 1,dc_Dims[1]=nmax*(nmax+2)+1+1;
  capi_dc_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_dc_tmp = array_from_pyobj(NPY_DOUBLE,dc_Dims,dc_Rank,capi_dc_intent,Py_None);
  if (capi_dc_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `dc' of uts_scsmfo.rotcoef to C/Fortran array" );
  } else {
    dc = (double *)(capi_dc_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&cbe,&kmax,&nmax,dc,&ndim);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_dc_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_dc_tmp == NULL) ... else of dc*/
  /* End of cleaning variable dc */
  } /*if (f2py_success) of kmax*/
  /* End of cleaning variable kmax */
  } /*if (f2py_success) of cbe*/
  /* End of cleaning variable cbe */
  } /*if (f2py_success) of ndim*/
  /* End of cleaning variable ndim */
  } /*if (f2py_success) of nmax*/
  /* End of cleaning variable nmax */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of rotcoef *******************************/

/*********************************** asmfr ***********************************/
static char doc_f2py_rout_uts_scsmfo_asmfr[] = "\
Function signature:\n\
  sa = asmfr(amn0,nodrt,theta,phi,kr)\n\
Required arguments:\n"
"  amn0 : input rank-3 array('D') with bounds (2,nodrt*(nodrt+2),2)\n"
"  nodrt : input int\n"
"  theta : input float\n"
"  phi : input float\n"
"  kr : input float\n"
"Return objects:\n"
"  sa : rank-1 array('D') with bounds (4)";
/* extern void F_FUNC(asmfr,ASMFR)(complex_double*,int*,double*,double*,double*,complex_double*); */
static PyObject *f2py_rout_uts_scsmfo_asmfr(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(complex_double*,int*,double*,double*,double*,complex_double*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  complex_double *amn0 = NULL;
  npy_intp amn0_Dims[3] = {-1, -1, -1};
  const int amn0_Rank = 3;
  PyArrayObject *capi_amn0_tmp = NULL;
  int capi_amn0_intent = 0;
  PyObject *amn0_capi = Py_None;
  int nodrt = 0;
  PyObject *nodrt_capi = Py_None;
  double theta = 0;
  PyObject *theta_capi = Py_None;
  double phi = 0;
  PyObject *phi_capi = Py_None;
  double kr = 0;
  PyObject *kr_capi = Py_None;
  complex_double *sa = NULL;
  npy_intp sa_Dims[1] = {-1};
  const int sa_Rank = 1;
  PyArrayObject *capi_sa_tmp = NULL;
  int capi_sa_intent = 0;
  static char *capi_kwlist[] = {"amn0","nodrt","theta","phi","kr",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOO:uts_scsmfo.asmfr",\
    capi_kwlist,&amn0_capi,&nodrt_capi,&theta_capi,&phi_capi,&kr_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable nodrt */
    f2py_success = int_from_pyobj(&nodrt,nodrt_capi,"uts_scsmfo.asmfr() 2nd argument (nodrt) can't be converted to int");
  if (f2py_success) {
  /* Processing variable phi */
    f2py_success = double_from_pyobj(&phi,phi_capi,"uts_scsmfo.asmfr() 4th argument (phi) can't be converted to double");
  if (f2py_success) {
  /* Processing variable kr */
    f2py_success = double_from_pyobj(&kr,kr_capi,"uts_scsmfo.asmfr() 5th argument (kr) can't be converted to double");
  if (f2py_success) {
  /* Processing variable theta */
    f2py_success = double_from_pyobj(&theta,theta_capi,"uts_scsmfo.asmfr() 3rd argument (theta) can't be converted to double");
  if (f2py_success) {
  /* Processing variable sa */
  sa_Dims[0]=4;
  capi_sa_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_sa_tmp = array_from_pyobj(NPY_CDOUBLE,sa_Dims,sa_Rank,capi_sa_intent,Py_None);
  if (capi_sa_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `sa' of uts_scsmfo.asmfr to C/Fortran array" );
  } else {
    sa = (complex_double *)(capi_sa_tmp->data);

  /* Processing variable amn0 */
  amn0_Dims[0]=2,amn0_Dims[1]=nodrt*(nodrt+2),amn0_Dims[2]=2;
  capi_amn0_intent |= F2PY_INTENT_IN;
  capi_amn0_tmp = array_from_pyobj(NPY_CDOUBLE,amn0_Dims,amn0_Rank,capi_amn0_intent,amn0_capi);
  if (capi_amn0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting 1st argument `amn0' of uts_scsmfo.asmfr to C/Fortran array" );
  } else {
    amn0 = (complex_double *)(capi_amn0_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(amn0,&nodrt,&theta,&phi,&kr,sa);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_sa_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_amn0_tmp!=amn0_capi) {
    Py_XDECREF(capi_amn0_tmp); }
  }  /*if (capi_amn0_tmp == NULL) ... else of amn0*/
  /* End of cleaning variable amn0 */
  }  /*if (capi_sa_tmp == NULL) ... else of sa*/
  /* End of cleaning variable sa */
  } /*if (f2py_success) of theta*/
  /* End of cleaning variable theta */
  } /*if (f2py_success) of kr*/
  /* End of cleaning variable kr */
  } /*if (f2py_success) of phi*/
  /* End of cleaning variable phi */
  } /*if (f2py_success) of nodrt*/
  /* End of cleaning variable nodrt */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************** end of asmfr ********************************/

/****************************** ms_radial_fields ******************************/
static char doc_f2py_rout_uts_scsmfo_ms_radial_fields[] = "\
Function signature:\n\
  as_rad = ms_radial_fields(amn0,nodrt,theta,phi,kr)\n\
Required arguments:\n"
"  amn0 : input rank-3 array('D') with bounds (2,nodrt*(nodrt+2),2)\n"
"  nodrt : input int\n"
"  theta : input float\n"
"  phi : input float\n"
"  kr : input float\n"
"Return objects:\n"
"  as_rad : rank-1 array('D') with bounds (2)";
/* extern void F_FUNC_US(ms_radial_fields,MS_RADIAL_FIELDS)(complex_double*,int*,double*,double*,double*,complex_double*); */
static PyObject *f2py_rout_uts_scsmfo_ms_radial_fields(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(complex_double*,int*,double*,double*,double*,complex_double*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  complex_double *amn0 = NULL;
  npy_intp amn0_Dims[3] = {-1, -1, -1};
  const int amn0_Rank = 3;
  PyArrayObject *capi_amn0_tmp = NULL;
  int capi_amn0_intent = 0;
  PyObject *amn0_capi = Py_None;
  int nodrt = 0;
  PyObject *nodrt_capi = Py_None;
  double theta = 0;
  PyObject *theta_capi = Py_None;
  double phi = 0;
  PyObject *phi_capi = Py_None;
  double kr = 0;
  PyObject *kr_capi = Py_None;
  complex_double *as_rad = NULL;
  npy_intp as_rad_Dims[1] = {-1};
  const int as_rad_Rank = 1;
  PyArrayObject *capi_as_rad_tmp = NULL;
  int capi_as_rad_intent = 0;
  static char *capi_kwlist[] = {"amn0","nodrt","theta","phi","kr",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOO:uts_scsmfo.ms_radial_fields",\
    capi_kwlist,&amn0_capi,&nodrt_capi,&theta_capi,&phi_capi,&kr_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable nodrt */
    f2py_success = int_from_pyobj(&nodrt,nodrt_capi,"uts_scsmfo.ms_radial_fields() 2nd argument (nodrt) can't be converted to int");
  if (f2py_success) {
  /* Processing variable phi */
    f2py_success = double_from_pyobj(&phi,phi_capi,"uts_scsmfo.ms_radial_fields() 4th argument (phi) can't be converted to double");
  if (f2py_success) {
  /* Processing variable kr */
    f2py_success = double_from_pyobj(&kr,kr_capi,"uts_scsmfo.ms_radial_fields() 5th argument (kr) can't be converted to double");
  if (f2py_success) {
  /* Processing variable as_rad */
  as_rad_Dims[0]=2;
  capi_as_rad_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_as_rad_tmp = array_from_pyobj(NPY_CDOUBLE,as_rad_Dims,as_rad_Rank,capi_as_rad_intent,Py_None);
  if (capi_as_rad_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `as_rad' of uts_scsmfo.ms_radial_fields to C/Fortran array" );
  } else {
    as_rad = (complex_double *)(capi_as_rad_tmp->data);

  /* Processing variable theta */
    f2py_success = double_from_pyobj(&theta,theta_capi,"uts_scsmfo.ms_radial_fields() 3rd argument (theta) can't be converted to double");
  if (f2py_success) {
  /* Processing variable amn0 */
  amn0_Dims[0]=2,amn0_Dims[1]=nodrt*(nodrt+2),amn0_Dims[2]=2;
  capi_amn0_intent |= F2PY_INTENT_IN;
  capi_amn0_tmp = array_from_pyobj(NPY_CDOUBLE,amn0_Dims,amn0_Rank,capi_amn0_intent,amn0_capi);
  if (capi_amn0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting 1st argument `amn0' of uts_scsmfo.ms_radial_fields to C/Fortran array" );
  } else {
    amn0 = (complex_double *)(capi_amn0_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(amn0,&nodrt,&theta,&phi,&kr,as_rad);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_as_rad_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_amn0_tmp!=amn0_capi) {
    Py_XDECREF(capi_amn0_tmp); }
  }  /*if (capi_amn0_tmp == NULL) ... else of amn0*/
  /* End of cleaning variable amn0 */
  } /*if (f2py_success) of theta*/
  /* End of cleaning variable theta */
  }  /*if (capi_as_rad_tmp == NULL) ... else of as_rad*/
  /* End of cleaning variable as_rad */
  } /*if (f2py_success) of kr*/
  /* End of cleaning variable kr */
  } /*if (f2py_success) of phi*/
  /* End of cleaning variable phi */
  } /*if (f2py_success) of nodrt*/
  /* End of cleaning variable nodrt */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/************************** end of ms_radial_fields **************************/

/************************************ asm ************************************/
static char doc_f2py_rout_uts_scsmfo_asm[] = "\
Function signature:\n\
  sa = asm(amn0,nodrt,theta,phi)\n\
Required arguments:\n"
"  amn0 : input rank-3 array('D') with bounds (2,nodrt*(nodrt+2),2)\n"
"  nodrt : input int\n"
"  theta : input float\n"
"  phi : input float\n"
"Return objects:\n"
"  sa : rank-1 array('D') with bounds (4)";
/* extern void F_FUNC(asm,ASM)(complex_double*,int*,double*,double*,complex_double*); */
static PyObject *f2py_rout_uts_scsmfo_asm(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(complex_double*,int*,double*,double*,complex_double*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  complex_double *amn0 = NULL;
  npy_intp amn0_Dims[3] = {-1, -1, -1};
  const int amn0_Rank = 3;
  PyArrayObject *capi_amn0_tmp = NULL;
  int capi_amn0_intent = 0;
  PyObject *amn0_capi = Py_None;
  int nodrt = 0;
  PyObject *nodrt_capi = Py_None;
  double theta = 0;
  PyObject *theta_capi = Py_None;
  double phi = 0;
  PyObject *phi_capi = Py_None;
  complex_double *sa = NULL;
  npy_intp sa_Dims[1] = {-1};
  const int sa_Rank = 1;
  PyArrayObject *capi_sa_tmp = NULL;
  int capi_sa_intent = 0;
  static char *capi_kwlist[] = {"amn0","nodrt","theta","phi",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOO:uts_scsmfo.asm",\
    capi_kwlist,&amn0_capi,&nodrt_capi,&theta_capi,&phi_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable phi */
    f2py_success = double_from_pyobj(&phi,phi_capi,"uts_scsmfo.asm() 4th argument (phi) can't be converted to double");
  if (f2py_success) {
  /* Processing variable nodrt */
    f2py_success = int_from_pyobj(&nodrt,nodrt_capi,"uts_scsmfo.asm() 2nd argument (nodrt) can't be converted to int");
  if (f2py_success) {
  /* Processing variable theta */
    f2py_success = double_from_pyobj(&theta,theta_capi,"uts_scsmfo.asm() 3rd argument (theta) can't be converted to double");
  if (f2py_success) {
  /* Processing variable sa */
  sa_Dims[0]=4;
  capi_sa_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_sa_tmp = array_from_pyobj(NPY_CDOUBLE,sa_Dims,sa_Rank,capi_sa_intent,Py_None);
  if (capi_sa_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `sa' of uts_scsmfo.asm to C/Fortran array" );
  } else {
    sa = (complex_double *)(capi_sa_tmp->data);

  /* Processing variable amn0 */
  amn0_Dims[0]=2,amn0_Dims[1]=nodrt*(nodrt+2),amn0_Dims[2]=2;
  capi_amn0_intent |= F2PY_INTENT_IN;
  capi_amn0_tmp = array_from_pyobj(NPY_CDOUBLE,amn0_Dims,amn0_Rank,capi_amn0_intent,amn0_capi);
  if (capi_amn0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting 1st argument `amn0' of uts_scsmfo.asm to C/Fortran array" );
  } else {
    amn0 = (complex_double *)(capi_amn0_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(amn0,&nodrt,&theta,&phi,sa);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_sa_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_amn0_tmp!=amn0_capi) {
    Py_XDECREF(capi_amn0_tmp); }
  }  /*if (capi_amn0_tmp == NULL) ... else of amn0*/
  /* End of cleaning variable amn0 */
  }  /*if (capi_sa_tmp == NULL) ... else of sa*/
  /* End of cleaning variable sa */
  } /*if (f2py_success) of theta*/
  /* End of cleaning variable theta */
  } /*if (f2py_success) of nodrt*/
  /* End of cleaning variable nodrt */
  } /*if (f2py_success) of phi*/
  /* End of cleaning variable phi */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/********************************* end of asm *********************************/

/*********************************** sbesjy ***********************************/
static char doc_f2py_rout_uts_scsmfo_sbesjy[] = "\
Function signature:\n\
  j,y,jp,yp,ifail = sbesjy(x,lmax)\n\
Required arguments:\n"
"  x : input float\n"
"  lmax : input int\n"
"Return objects:\n"
"  j : rank-1 array('d') with bounds (lmax + 1)\n"
"  y : rank-1 array('d') with bounds (lmax + 1)\n"
"  jp : rank-1 array('d') with bounds (lmax + 1)\n"
"  yp : rank-1 array('d') with bounds (lmax + 1)\n"
"  ifail : int";
/* extern void F_FUNC(sbesjy,SBESJY)(double*,int*,double*,double*,double*,double*,int*); */
static PyObject *f2py_rout_uts_scsmfo_sbesjy(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,int*,double*,double*,double*,double*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double x = 0;
  PyObject *x_capi = Py_None;
  int lmax = 0;
  PyObject *lmax_capi = Py_None;
  double *j = NULL;
  npy_intp j_Dims[1] = {-1};
  const int j_Rank = 1;
  PyArrayObject *capi_j_tmp = NULL;
  int capi_j_intent = 0;
  double *y = NULL;
  npy_intp y_Dims[1] = {-1};
  const int y_Rank = 1;
  PyArrayObject *capi_y_tmp = NULL;
  int capi_y_intent = 0;
  double *jp = NULL;
  npy_intp jp_Dims[1] = {-1};
  const int jp_Rank = 1;
  PyArrayObject *capi_jp_tmp = NULL;
  int capi_jp_intent = 0;
  double *yp = NULL;
  npy_intp yp_Dims[1] = {-1};
  const int yp_Rank = 1;
  PyArrayObject *capi_yp_tmp = NULL;
  int capi_yp_intent = 0;
  int ifail = 0;
  static char *capi_kwlist[] = {"x","lmax",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO:uts_scsmfo.sbesjy",\
    capi_kwlist,&x_capi,&lmax_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable ifail */
  /* Processing variable lmax */
    f2py_success = int_from_pyobj(&lmax,lmax_capi,"uts_scsmfo.sbesjy() 2nd argument (lmax) can't be converted to int");
  if (f2py_success) {
  /* Processing variable x */
    f2py_success = double_from_pyobj(&x,x_capi,"uts_scsmfo.sbesjy() 1st argument (x) can't be converted to double");
  if (f2py_success) {
  /* Processing variable jp */
  jp_Dims[0]=lmax + 1;
  capi_jp_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_jp_tmp = array_from_pyobj(NPY_DOUBLE,jp_Dims,jp_Rank,capi_jp_intent,Py_None);
  if (capi_jp_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `jp' of uts_scsmfo.sbesjy to C/Fortran array" );
  } else {
    jp = (double *)(capi_jp_tmp->data);

  /* Processing variable yp */
  yp_Dims[0]=lmax + 1;
  capi_yp_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_yp_tmp = array_from_pyobj(NPY_DOUBLE,yp_Dims,yp_Rank,capi_yp_intent,Py_None);
  if (capi_yp_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `yp' of uts_scsmfo.sbesjy to C/Fortran array" );
  } else {
    yp = (double *)(capi_yp_tmp->data);

  /* Processing variable j */
  j_Dims[0]=lmax + 1;
  capi_j_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_j_tmp = array_from_pyobj(NPY_DOUBLE,j_Dims,j_Rank,capi_j_intent,Py_None);
  if (capi_j_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `j' of uts_scsmfo.sbesjy to C/Fortran array" );
  } else {
    j = (double *)(capi_j_tmp->data);

  /* Processing variable y */
  y_Dims[0]=lmax + 1;
  capi_y_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_y_tmp = array_from_pyobj(NPY_DOUBLE,y_Dims,y_Rank,capi_y_intent,Py_None);
  if (capi_y_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(uts_scsmfo_error,"failed in converting hidden `y' of uts_scsmfo.sbesjy to C/Fortran array" );
  } else {
    y = (double *)(capi_y_tmp->data);

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&x,&lmax,j,y,jp,yp,&ifail);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NNNNi",capi_j_tmp,capi_y_tmp,capi_jp_tmp,capi_yp_tmp,ifail);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_y_tmp == NULL) ... else of y*/
  /* End of cleaning variable y */
  }  /*if (capi_j_tmp == NULL) ... else of j*/
  /* End of cleaning variable j */
  }  /*if (capi_yp_tmp == NULL) ... else of yp*/
  /* End of cleaning variable yp */
  }  /*if (capi_jp_tmp == NULL) ... else of jp*/
  /* End of cleaning variable jp */
  } /*if (f2py_success) of x*/
  /* End of cleaning variable x */
  } /*if (f2py_success) of lmax*/
  /* End of cleaning variable lmax */
  /* End of cleaning variable ifail */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of sbesjy *******************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"rotcoef",-1,{{-1}},0,(char *)F_FUNC(rotcoef,ROTCOEF),(f2py_init_func)f2py_rout_uts_scsmfo_rotcoef,doc_f2py_rout_uts_scsmfo_rotcoef},
  {"asmfr",-1,{{-1}},0,(char *)F_FUNC(asmfr,ASMFR),(f2py_init_func)f2py_rout_uts_scsmfo_asmfr,doc_f2py_rout_uts_scsmfo_asmfr},
  {"ms_radial_fields",-1,{{-1}},0,(char *)F_FUNC_US(ms_radial_fields,MS_RADIAL_FIELDS),(f2py_init_func)f2py_rout_uts_scsmfo_ms_radial_fields,doc_f2py_rout_uts_scsmfo_ms_radial_fields},
  {"asm",-1,{{-1}},0,(char *)F_FUNC(asm,ASM),(f2py_init_func)f2py_rout_uts_scsmfo_asm,doc_f2py_rout_uts_scsmfo_asm},
  {"sbesjy",-1,{{-1}},0,(char *)F_FUNC(sbesjy,SBESJY),(f2py_init_func)f2py_rout_uts_scsmfo_sbesjy,doc_f2py_rout_uts_scsmfo_sbesjy},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "uts_scsmfo",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyObject *PyInit_uts_scsmfo(void) {
#else
#define RETVAL
PyMODINIT_FUNC inituts_scsmfo(void) {
#endif
  int i;
  PyObject *m,*d, *s;
#if PY_VERSION_HEX >= 0x03000000
  m = uts_scsmfo_module = PyModule_Create(&moduledef);
#else
  m = uts_scsmfo_module = Py_InitModule("uts_scsmfo", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module uts_scsmfo (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module 'uts_scsmfo' is auto-generated with f2py (version:2).\nFunctions:\n"
"  dc = rotcoef(cbe,kmax,nmax,ndim)\n"
"  sa = asmfr(amn0,nodrt,theta,phi,kr)\n"
"  as_rad = ms_radial_fields(amn0,nodrt,theta,phi,kr)\n"
"  sa = asm(amn0,nodrt,theta,phi)\n"
"  j,y,jp,yp,ifail = sbesjy(x,lmax)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  uts_scsmfo_error = PyErr_NewException ("uts_scsmfo.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));





/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"uts_scsmfo");
#endif

  return RETVAL;
}
#ifdef __cplusplus
}
#endif
