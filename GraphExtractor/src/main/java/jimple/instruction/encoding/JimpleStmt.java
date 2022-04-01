package jimple.instruction.encoding;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.json.JSONObject;
import soot.Local;
import soot.Modifier;
import soot.Scene;
import soot.SootField;
import soot.SootMethod;
import soot.SootMethodRef;
import soot.Unit;
import soot.Value;
import soot.dexpler.typing.UntypedConstant;
import soot.jimple.AddExpr;
import soot.jimple.AndExpr;
import soot.jimple.AnyNewExpr;
import soot.jimple.ArrayRef;
import soot.jimple.BinopExpr;
import soot.jimple.CastExpr;
import soot.jimple.CaughtExceptionRef;
import soot.jimple.ClassConstant;
import soot.jimple.CmpExpr;
import soot.jimple.CmpgExpr;
import soot.jimple.CmplExpr;
import soot.jimple.ConcreteRef;
import soot.jimple.Constant;
import soot.jimple.DivExpr;
import soot.jimple.DoubleConstant;
import soot.jimple.DynamicInvokeExpr;
import soot.jimple.EqExpr;
import soot.jimple.Expr;
import soot.jimple.FieldRef;
import soot.jimple.FloatConstant;
import soot.jimple.GeExpr;
import soot.jimple.GtExpr;
import soot.jimple.IdentityRef;
import soot.jimple.InstanceFieldRef;
import soot.jimple.InstanceOfExpr;
import soot.jimple.IntConstant;
import soot.jimple.InvokeExpr;
import soot.jimple.LeExpr;
import soot.jimple.LongConstant;
import soot.jimple.LtExpr;
import soot.jimple.MethodHandle;
import soot.jimple.MethodType;
import soot.jimple.MulExpr;
import soot.jimple.NeExpr;
import soot.jimple.NewArrayExpr;
import soot.jimple.NewExpr;
import soot.jimple.NewMultiArrayExpr;
import soot.jimple.NullConstant;
import soot.jimple.NumericConstant;
import soot.jimple.OrExpr;
import soot.jimple.ParameterRef;
import soot.jimple.RemExpr;
import soot.jimple.ShlExpr;
import soot.jimple.ShrExpr;
import soot.jimple.StaticFieldRef;
import soot.jimple.StringConstant;
import soot.jimple.SubExpr;
import soot.jimple.ThisRef;
import soot.jimple.UnopExpr;
import soot.jimple.UshrExpr;
import soot.jimple.XorExpr;
import soot.jimple.internal.AbstractCastExpr;
import soot.jimple.internal.AbstractInstanceFieldRef;
import soot.jimple.internal.AbstractInstanceOfExpr;
import soot.jimple.internal.AbstractInterfaceInvokeExpr;
import soot.jimple.internal.AbstractLengthExpr;
import soot.jimple.internal.AbstractNegExpr;
import soot.jimple.internal.AbstractNewArrayExpr;
import soot.jimple.internal.AbstractNewExpr;
import soot.jimple.internal.AbstractNewMultiArrayExpr;
import soot.jimple.internal.AbstractSpecialInvokeExpr;
import soot.jimple.internal.AbstractStaticInvokeExpr;
import soot.jimple.internal.AbstractVirtualInvokeExpr;
import soot.jimple.internal.JAssignStmt;
import soot.jimple.internal.JBreakpointStmt;
import soot.jimple.internal.JCastExpr;
import soot.jimple.internal.JCaughtExceptionRef;
import soot.jimple.internal.JDynamicInvokeExpr;
import soot.jimple.internal.JEnterMonitorStmt;
import soot.jimple.internal.JExitMonitorStmt;
import soot.jimple.internal.JGotoStmt;
import soot.jimple.internal.JIdentityStmt;
import soot.jimple.internal.JIfStmt;
import soot.jimple.internal.JInstanceOfExpr;
import soot.jimple.internal.JInvokeStmt;
import soot.jimple.internal.JLookupSwitchStmt;
import soot.jimple.internal.JNopStmt;
import soot.jimple.internal.JRetStmt;
import soot.jimple.internal.JReturnStmt;
import soot.jimple.internal.JReturnVoidStmt;
import soot.jimple.internal.JTableSwitchStmt;
import soot.jimple.internal.JThrowStmt;
import soot.shimple.toolkits.scalar.SEvaluator.BottomConstant;
import soot.shimple.toolkits.scalar.SEvaluator.TopConstant;
import soot.Type;
import jimple.io.Common;

public class JimpleStmt{
	ArrayList<Unit> visitedlist = new ArrayList<Unit>();
	public void cleanVisitedList() {
		this.visitedlist.clear();
	}
    public JSONObject toJson(Unit u) throws Exception {
    	if (visitedlist.contains(u)) {
    		JSONObject ob = new JSONObject();
    		ob.put("Point", "Head");
    		return ob;
    	}else {
    		visitedlist.add(u);
    	}
		if( u instanceof JAssignStmt) {
			return this.toJson((JAssignStmt)u);
		}else if( u instanceof JIdentityStmt) {
		    return this.toJson((JIdentityStmt)u);	
		}else if( u instanceof JEnterMonitorStmt) {
			return this.toJson((JEnterMonitorStmt)u);
		}else if( u instanceof JExitMonitorStmt) {
			return this.toJson((JExitMonitorStmt)u);
		}else if( u instanceof JReturnStmt) {
			return this.toJson((JReturnStmt)u);
		}else if( u instanceof JThrowStmt) {
			return this.toJson((JThrowStmt)u);
		}else if( u instanceof JLookupSwitchStmt) {
			return this.toJson((JLookupSwitchStmt)u);
		}else if( u instanceof JTableSwitchStmt) {
			return this.toJson((JTableSwitchStmt)u);
		}else if( u instanceof JBreakpointStmt) {
			return this.toJson((JBreakpointStmt)u);
		}else if( u instanceof JGotoStmt) {
			return this.toJson((JGotoStmt)u);
		}else if( u instanceof JIfStmt) {
			return this.toJson((JIfStmt)u);
		}else if ( u instanceof JInvokeStmt) {
			return this.toJson((JInvokeStmt)u);
		}else if ( u instanceof  JNopStmt) {
			return this.toJson((JNopStmt)u);
		}else if( u instanceof JRetStmt) {
			return this.toJson((JRetStmt)u);
		}else if( u instanceof JReturnVoidStmt) {
			return this.toJson((JReturnVoidStmt)u);
		}else {
			throw new Exception(u.toString()+" Invalid Jimple");
		}
	}
	

	public JSONObject toJson(JAssignStmt u) throws Exception {
		JSONObject jobject = new JSONObject();
		JSONObject leftaddr = this.toJson(u.getLeftOp());
		JSONObject rightaddr = this.toJson(u.getRightOp());
		
		jobject.put("Jimple", "AssignStmt");
		jobject.put("left", leftaddr);
		jobject.put("right", rightaddr);
		//if(rightaddr.isEmpty()) {
		
		  //leftaddr = this.toJson(u.getLeftOp());
		  //rightaddr = this.toJson(u.getRightOp());
		  //}
		return jobject;
	}
	
	public JSONObject toJson(JIdentityStmt u) throws Exception {
		JSONObject jobject = new JSONObject();
		JSONObject leftaddr = this.toJson(u.getLeftOp());
		JSONObject rightaddr = this.toJson(u.getRightOp());
		jobject.put("Jimple", "IdentityStmt");
		jobject.put("left", leftaddr);
		jobject.put("right", rightaddr);
		return jobject;
		}
	
	public JSONObject toJson(JEnterMonitorStmt u) throws Exception{
		JSONObject jobject = new JSONObject();
		jobject.put("Jimple", "EnterMonitorStmt");
		jobject.put("value", this.toJson(u.getOp()));
		return jobject;
		
	}
	
	public JSONObject toJson(JExitMonitorStmt u) throws Exception{
		JSONObject jobject = new JSONObject();
		jobject.put("Jimple", "ExitMonitorStmt");
		jobject.put("value", this.toJson(u.getOp()));
		return jobject;
	}
	
	public JSONObject toJson(JReturnStmt u) throws Exception{
		JSONObject jobject = new JSONObject();
		jobject.put("Jimple", "ReturnStmt");
		jobject.put("value", this.toJson(u.getOp()));
		return jobject;
	}
	
	public JSONObject toJson(JThrowStmt u) throws Exception{
		JSONObject jobject = new JSONObject();
		jobject.put("Jimple", "ThrowStmt");
		jobject.put("value", this.toJson(u.getOp()));
		return jobject;
	}
	
	
	public JSONObject toJson(JLookupSwitchStmt j) throws Exception{
		JSONObject ojson = new JSONObject();
		JSONObject keyValue = this.toJson(j.getKey());
		JSONObject options = new JSONObject();
	    for (int i = 0; i < j.getLookupValues().size(); i++) {
			Unit target = j.getTarget(i);
			int casevalue = j.getLookupValue(i);
		    JSONObject ut = this.toJson(target);
		    options.put(Integer.toString(casevalue), ut);
		}

	    Unit target = j.getDefaultTarget();
	    JSONObject dt = this.toJson(target);
	    options.put("Default", dt);
	    ojson.put("Jimple", "LookupSwitchStmt");
	    ojson.put("key", keyValue);
	    ojson.put("value", options);
		return ojson;
	}
	
	public JSONObject toJson(JTableSwitchStmt j) throws Exception{
		JSONObject ojson = new JSONObject();
		JSONObject keyValue = this.toJson(j.getKey());
		int lowIndex = j.getLowIndex();
		int highIndex = j.getHighIndex();
		JSONObject options = new JSONObject();
	    for (int i = lowIndex; i < highIndex; i++) {
	      Unit target = j.getTarget(i - lowIndex);
		  JSONObject ut = this.toJson(target);
		  options.put(Integer.toString(i), ut);
	    }
	    Unit target = j.getTarget(highIndex - lowIndex);
	    options.put(Integer.toString(highIndex), this.toJson(target));
	   
	    target = j.getDefaultTarget();
	    JSONObject dt = this.toJson(target);
	    options.put("Default", dt);
	    ojson.put("Jimple", "TableSwitchStmt");
	    ojson.put("key", keyValue);
	    ojson.put("value", options);
		return ojson;
		
	}
	
	public JSONObject toJson(JBreakpointStmt b) throws Exception{
		JSONObject ojson = new JSONObject();
		ojson.put("Jimple", "BreakpointStmt");
		return ojson;
	}
	
	public JSONObject toJson(JGotoStmt j) throws Exception{
		Unit target =  j.getTarget();
		JSONObject ob = this.toJson(target);
		JSONObject ojson = new JSONObject();
		ojson.put("Jimple", "GotoStmt");
		ojson.put("value", ob);
		return ojson;
		
	}
	
	public JSONObject toJson(JIfStmt j) throws Exception{
		Unit target = j.getTarget();
		Value condition = j.getCondition();
		JSONObject ojson = new JSONObject();
		ojson.put("target", this.toJson(target));
		ojson.put("value", this.toJson(condition));
		ojson.put("Jimple", "IfStmt");
		return ojson;
	}
	
	public JSONObject toJson(JInvokeStmt j) throws Exception{
		Value v = j.getInvokeExprBox().getValue();
		JSONObject invoke = this.toJson(v);
		JSONObject ojson = new JSONObject();
		ojson.put("Jimple", "InvokeStmt");
		ojson.put("value", invoke);
		return ojson;
	}
	
	public JSONObject toJson(JNopStmt j) throws Exception{
		JSONObject ojson = new JSONObject();
		ojson.put("Jimple", "NopStmt");
		return ojson;
	}
	
	public JSONObject toJson(JRetStmt j) throws Exception{
		JSONObject ojson = new JSONObject();
		JSONObject ob = this.toJson(j.getStmtAddressBox().getValue());
		ojson.put("Jimple", "RetStmt");
		ojson.put("value", ob);
		return ojson;
	}
	
	public JSONObject toJson(JReturnVoidStmt j) throws Exception{
		JSONObject ojson = new JSONObject();
		ojson.put("Jimple", "ReturnVoidStmt");
		return ojson;
	}
	
	public JSONObject toJson(Value v) throws Exception {
		JSONObject info = new JSONObject();
		if( v instanceof Local ) {
			Local l = (Local) v;
			String typename = l.getType().toString();
			String variablename = l.getName();
			info.put("Jimple", "Local");
			info.put("Type", typename);
			info.put("Varaible", variablename);
		} else if ( v instanceof ConcreteRef) {
			if( v instanceof ArrayRef ) {
				ArrayRef l = (ArrayRef) v;
				String typename = l.getType().toString();
				String variablename = l.getBase().toString();
				String indexname = l.getIndex().toString();
				info.put("Jimple", "ArrayRef");
				info.put("Type", typename);
				info.put("Varaible", variablename);
				info.put("Index", indexname);
			}else if ( v instanceof FieldRef ) {
				if ( v instanceof InstanceFieldRef) {
					AbstractInstanceFieldRef l = (AbstractInstanceFieldRef)v;
					String typename = l.getType().toString();
				    String basename = l.getBase().toString();
				    String signature = "";
				    String name = "";
					if(l.getField()!=null) {
						signature = l.getField().getSignature();
						name = l.getField().getName();
					}
				     
				    info.put("Jimple", "InstanceFieldRef");
				    info.put("Type", typename);
				    info.put("Basename", basename);
				    info.put("name", name);
				    info.put("Signature", signature);
				}else if( v instanceof StaticFieldRef) {
					StaticFieldRef l = (StaticFieldRef)v;
					String typename = l.getType().toString();
					String signature = "";
					if(l.getField()!=null) {
						signature = l.getField().getSignature();
					}
					info.put("Jimple", "StaticFieldRef");
				    info.put("Type", typename);
				    info.put("Signature", signature);
				}else {
					throw new Exception( v.toString() 
							+ " The value is not FieldRef .\n");
				}
			}
			
		} else if( v instanceof IdentityRef) {
			if( v instanceof ParameterRef ) {
				ParameterRef vp = (ParameterRef)v;
				String type = vp.getType().toString();
				String name = "@parameter: " + vp.getIndex();
				info.put("Jimple", "ParameterRef");
				info.put("Variable", name);
				info.put("Type", type);
			}else if( v instanceof ThisRef) {
				ThisRef vp = (ThisRef)v;
				String type = vp.getType().toString();
				String name = "@this";
				info.put("Jimple", "ThisRef");
				info.put("Type", type);
				info.put("Variable", name);
			}else if( v instanceof CaughtExceptionRef) {
				JCaughtExceptionRef vp = (JCaughtExceptionRef)v;
				String type = vp.getType().toString();
				String name = vp.toString();
				info.put("Jimple", "CaughtExceptionRef");
				info.put("Type", type);
				info.put("Variable", name);
			}else {
				throw new Exception( v.toString() 
						+ "  is not IdentityRef .\n");
			}
		}else if(v instanceof Constant) {
			info = this.toJson((Constant)v);
		}else if ( v instanceof Expr) {
			if ( v instanceof AnyNewExpr ) {
				info = this.toJson((AnyNewExpr)v);
			}else if( v instanceof BinopExpr) {
				info = this.toJson((BinopExpr)v);
			}else if( v instanceof CastExpr ) {
				info = this.toJson((CastExpr)v);
			}else if( v instanceof UnopExpr) {
				info = this.toJson((UnopExpr)v);
			}else if( v instanceof InstanceOfExpr) {
				info = this.toJson((InstanceOfExpr)v);
			}else if( v instanceof InvokeExpr) {
				info = this.toJson((InvokeExpr)v);
			}else {
				throw new Exception( v.toString() 
						+ " The value is not Expr .\n");
			}
		}
		else{
			throw new Exception( v.toString() 
					+ " The value is not value .\n");
		}
		return info;
		
	}
	
	public JSONObject toJson(Constant c) throws Exception {
		JSONObject ob = new JSONObject();
		if( c instanceof ClassConstant) {
			ClassConstant l = (ClassConstant)c;
			String type = l.toSootType().toString();
			ob.put("Jimple", "ClassConstant");
			ob.put("Type", type);
		}else if( c instanceof BottomConstant) {
			BottomConstant l = (BottomConstant)c;
			ob.put("Jimple", "BottomConstant");
		}else if( c instanceof TopConstant) {
			TopConstant l = (TopConstant)c;
			ob.put("Jimple", "TopConstant");
		}else if( c instanceof MethodHandle) {
			MethodHandle l = (MethodHandle)c;
		    String type = l.getKindString();
		    JSONObject methodinfo = this.toJson(l.getMethodRef());
		    JSONObject fieldinfo = new JSONObject();
		    if(l != null && l.getFieldRef() != null) {
		    	 fieldinfo = this.toJson(l.getFieldRef().resolve());
		    	}
		   
		    ob.put("Jimple", "MethodHandle");
			ob.put("Type", type);
			ob.put("method", methodinfo);
			ob.put("field", fieldinfo);
		}else if( c instanceof MethodType) {
			MethodType l = (MethodType)c;
			String jimpleType = l.getType().toString();
			String retrunType = l.getReturnType().toString();
			JSONObject arglist = new JSONObject();
			int counter = 0;
			for(Type t: l.getParameterTypes()) {
				arglist.put(Integer.toString(counter), t.toString());
				counter ++;
			}
			ob.put("Jimple", "MethodType");
			ob.put("retrun", retrunType);
			ob.put("params", arglist);
		}else if( c instanceof NullConstant) {
			ob.put("Jimple", "NullConstant");
		}else if( c instanceof NumericConstant) {
			if( c instanceof IntConstant ) {
				ob.put("Jimple", "IntConstant");
				ob.put("value", Integer.toString(((IntConstant)c).value));
			}else if ( c instanceof LongConstant ){
				ob.put("Jimple", "LongConstant");
				ob.put("value", Long.toString(((LongConstant)c).value));
			}else if ( c instanceof DoubleConstant ) {
				ob.put("Jimple", "DoubleConstant");
				ob.put("value", Double.toString(((DoubleConstant)c).value));
			}else if( c instanceof FloatConstant ) {
				ob.put("Jimple", "FloatConstant");
				ob.put("value", Float.toString(((FloatConstant)c).value));
			}else {
				throw new Exception( c.toString() 
						+ "  is not NumericConstant.\n");
			}
		}else if( c instanceof StringConstant ) {
			ob.put("Jimple", "StringConstant");
			ob.put("value", ((StringConstant)c).value);
		}else if( c instanceof UntypedConstant) {
			ob.put("Jimple", "UntypedConstant");
		}else {
			throw new Exception( c.toString() 
					+ "  is not Constant.\n");
		}
		return ob;
	}
	
	public JSONObject toJson(AnyNewExpr v) throws Exception{
		JSONObject ob = new JSONObject();
		if( v instanceof NewArrayExpr ) {
			AbstractNewArrayExpr l = (AbstractNewArrayExpr)v;
			String typename = l.getBaseType().toString();
			String shape = "["+l.getSize().toString()+"]";
			ob.append("Jimple", "NewArrayExpr");
			ob.append("Type", typename);
			ob.append("Dimension", "1");
			ob.append("shape", shape);
		}else if( v instanceof NewMultiArrayExpr) {
			AbstractNewMultiArrayExpr l = (AbstractNewMultiArrayExpr)v;
			String typename = l.getBaseType().toString();
			String dimenson = Integer.toString(l.getBaseType().numDimensions);
			StringBuffer buffer = new StringBuffer();
			for (Value element : l.getSizes()) {
			      buffer.append("[" + element.toString() + "]");
			    }
			String shape = buffer.toString();
			ob.append("Jimple", "NewMultiArrayExpr");
			ob.append("Type", typename);
			ob.append("Dimension", dimenson);
			ob.append("shape", shape);
		}else if( v instanceof NewExpr) {
			AbstractNewExpr l = (AbstractNewExpr)v;
			String typename = l.getBaseType().toString();
			ob.append("Jimple", "NewExpr");
			ob.append("Type", typename);
		}else {
			throw new Exception( v.toString() 
					+ " The left value of AnyNewExpr is not one of Local, ArrayRef and FieldRef.\n");
		}
		return ob;
	}
	
	public JSONObject toJson(BinopExpr v) throws Exception {
		JSONObject ob = new JSONObject();
		if( v instanceof AddExpr) {
			ob.put("Jimple", "Add");
		}else if( v instanceof AndExpr) {
			ob.put("Jimple", "And");
		}else if( v instanceof CmpExpr) {
			ob.put("Jimple", "Compare Equal");
		}else if( v instanceof CmpgExpr) {
			ob.put("Jimple", "Compare Greater");
		}else if( v instanceof CmplExpr) {
			ob.put("Jimple", "Compare Less");
		}else if( v instanceof EqExpr) {
			ob.put("Jimple", "Equal");
		}else if( v instanceof GeExpr) {
			ob.put("Jimple", "Greater Equal");
		}else if( v instanceof GtExpr) {
			ob.put("Jimple", "Greater Than");
		}else if( v instanceof LeExpr) {
			ob.put("Jimple", "Less Equal");
		}else if( v instanceof LtExpr) {
			ob.put("Jimple", "Less Than");
		}else if( v instanceof NeExpr) {
			ob.put("Jimple", "Not Equal");
		}else if( v instanceof DivExpr) {
			ob.put("Jimple", "Divide");
		}else if ( v instanceof MulExpr) {
			ob.put("Jimple", "Multiply");
		}else if( v instanceof OrExpr) {
			ob.put("Jimple", "Or");
		}else if( v instanceof RemExpr) {
			ob.put("Jimple", "Modulo");
		}else if( v instanceof ShlExpr) {
			ob.put("Jimple", "Shift Left");
		}else if( v instanceof ShrExpr) {
			ob.put("Jimple", "Shitf Right");
		}else if( v instanceof SubExpr) {
			ob.put("Jimple", "Sub");
		}else if( v instanceof UshrExpr) {
			ob.put("Jimple", "Ignoring Sign Shift Right");
		}else if( v instanceof XorExpr) {
			ob.put("Jimple", "Exclusive Or");
		}else {
			throw new Exception( v.toString() 
					+ "  is not BinopExpr.\n");
		}
		JSONObject op1 = this.toJson(v.getOp1());
		JSONObject op2 = this.toJson(v.getOp2());
		ob.put("op1", op1);
		ob.put("op2", op2);
		return ob;
	}
    
	public JSONObject toJson(CastExpr v) throws Exception{
		JSONObject ob = new JSONObject();
		if( v instanceof JCastExpr ) {
			AbstractCastExpr l = (AbstractCastExpr)v;
			String castTotype = l.getType().toString();
			JSONObject orgType = this.toJson(l.getOpBox().getValue());
			ob.put("Jimple", "CastExpr");
			ob.put("Type", castTotype);
			ob.put("op1", orgType);
		}else {
			throw new Exception( v.toString() 
					+ " is not JCastExpr.\n");
		}
		return ob;
	}
	
	public JSONObject toJson(UnopExpr v) throws Exception {
		JSONObject ob = new JSONObject();
		if( v instanceof AbstractLengthExpr) {
			AbstractLengthExpr l = (AbstractLengthExpr)v;
			JSONObject value = this.toJson(l.getOp());
			ob.put("Jimple", "LengthExpr");
			ob.put("op1", value);
		}else if( v instanceof AbstractNegExpr) {
			AbstractNegExpr l = (AbstractNegExpr)v;
			JSONObject value = this.toJson(l.getOp());
			ob.put("Jimple", "NegExpr");
			ob.put("op1", value);	
		}else {
			throw new Exception( v.toString() 
					+ " is not UnopExpr.\n");
		}
		return ob;
	}
	
	public JSONObject toJson(InstanceOfExpr v) throws Exception{
		JSONObject ob = new JSONObject();
		if(v instanceof JInstanceOfExpr ) {
			AbstractInstanceOfExpr l = (AbstractInstanceOfExpr)v;
			JSONObject op1 = this.toJson(l.getOp());
			String checktype = l.getCheckType().toString();
			ob.put("Jimple", "InstanceOfExpr");
			ob.put("op1", op1);
			ob.put("Type", checktype);
		}else {
			throw new Exception( v.toString() 
					+ " is not JInstanceOfExpr.\n");
		}
		return ob;	
	}
	
	public JSONObject toJson(InvokeExpr v) throws Exception {
		JSONObject ob = new JSONObject();
		if( v instanceof AbstractInterfaceInvokeExpr ) {
			AbstractInterfaceInvokeExpr l = (AbstractInterfaceInvokeExpr)v;
			JSONObject op1 = this.toJson(l.getBase());
			JSONObject methodinfo = this.toJson(l.getMethodRef());
			JSONObject argslist = new JSONObject();
			List<Value> lvs = l.getArgs();
			int counter = 0;
			for(Value arg : lvs) {
				if( arg != null) {
					argslist.put(Integer.toString(counter), this.toJson(arg));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					argslist.put(Integer.toString(counter), njson);
				}
				counter++;	
			}
			ob.put("Jimple", "InterfaceInvokeExpr");
			ob.put("op1", op1);
			ob.put("method", methodinfo);
			ob.put("args", argslist);
		}else if ( v instanceof AbstractSpecialInvokeExpr) {
			AbstractSpecialInvokeExpr l = (AbstractSpecialInvokeExpr)v;
			JSONObject op1 = this.toJson(l.getBase());
			JSONObject methodinfo = this.toJson(l.getMethodRef());
			JSONObject argslist = new JSONObject();
			List<Value> lvs = l.getArgs();
			int counter = 0;
			for(Value arg : lvs) {
				if( arg != null) {
					argslist.put(Integer.toString(counter), this.toJson(arg));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					argslist.put(Integer.toString(counter), njson);
				}
				counter++;	
			}
			ob.put("Jimple", "SpecialInvokeExpr");
			ob.put("op1", op1);
			ob.put("method", methodinfo);
			ob.put("args", argslist);
		}else if( v instanceof AbstractVirtualInvokeExpr) {
			AbstractVirtualInvokeExpr l = (AbstractVirtualInvokeExpr)v;
			JSONObject op1 = this.toJson(l.getBase());
			JSONObject methodinfo = this.toJson(l.getMethodRef());
			JSONObject argslist = new JSONObject();
			List<Value> lvs = l.getArgs();
			int counter = 0;
			for(Value arg : lvs) {
				if( arg != null) {
					argslist.put(Integer.toString(counter), this.toJson(arg));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					argslist.put(Integer.toString(counter), njson);
				}
				counter++;	
			}
			ob.put("Jimple", "VirtualInvokeExpr");
			ob.put("op1", op1);
			ob.put("method", methodinfo);
			ob.put("args", argslist);
		}else if( v instanceof AbstractStaticInvokeExpr) {
			AbstractStaticInvokeExpr l = (AbstractStaticInvokeExpr)v;
			JSONObject methodinfo = this.toJson(l.getMethodRef());
			JSONObject argslist = new JSONObject();
			List<Value> lvs = l.getArgs();
			int counter = 0;
			for(Value arg : lvs) {
				if( arg != null) {
					argslist.put(Integer.toString(counter), this.toJson(arg));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					argslist.put(Integer.toString(counter), njson);
				}
				counter++;	
			}
			ob.put("Jimple", "StaticInvokeExpr");
			ob.put("method", methodinfo);
			ob.put("args", argslist);
		}else if( v instanceof DynamicInvokeExpr) {
			JDynamicInvokeExpr l = (JDynamicInvokeExpr)v;
			String dynamicname = SootMethod.getSubSignature(""/* no method name here */, 
					l.getMethodRef().parameterTypes(), l.getMethodRef().returnType());
			List<Value> args = l.getArgs();
			JSONObject argslist = new JSONObject();
			int counter = 0;
			for( Value arg : args) {
				if( arg != null) {
					argslist.put(Integer.toString(counter), this.toJson(arg));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					argslist.put(Integer.toString(counter), njson);
				}
			}
			JSONObject boostJobject = new JSONObject();
			counter = 0;
			List<Value> boostValues = l.getBootstrapArgs();
			for( Value bv: boostValues) {
				if(bv != null ) {
				    boostJobject.put(Integer.toString(counter), this.toJson(bv));
				}else {
					JSONObject njson = new JSONObject();
					njson.put("Null", "null");
					boostJobject.put(Integer.toString(counter), njson);
				}
			}
			ob.put("Jimple", "DynamicInvokeExpr");
			ob.put("boost", boostJobject);
			ob.put("args", argslist);
		}else {
			throw new Exception( v.toString() 
					+ " The Jimple cmd is not JCastExpr.\n");
		}
		return ob;
		
	}
	
	public JSONObject toJson(SootMethodRef v) {
		JSONObject ob = new JSONObject();
		String classname = v.getDeclaringClass().toString();
		String signature = v.getSignature();
		String returnName = v.getReturnType().toString();
		String methodname = v.getName();
		JSONObject paramType = new JSONObject();
		int counter = 0;
		for(Type t: v.getParameterTypes()) {
			paramType.put(Integer.toString(counter), t.toString());
			counter++;
		}
		ob.put("Jimple", "SootMethodRef");
		ob.put("class", classname);
		ob.put("name", methodname);
		
		String normalizedMethodName = Common.normalizeName(methodname, Common.BlankWord);
		ArrayList<String> splitNameParts = Common.splitToSubtokens(methodname);
		String splitName = normalizedMethodName;
		if (splitNameParts.size() > 0) {
			splitName = splitNameParts.stream().collect(Collectors.joining(Common.internalSeparator));
		}
		ob.put("normalized_name", splitName);
		ob.put("signature", signature);
		ob.put("return", returnName);
		ob.put("params", paramType);
		return ob;
		
	}
	
	public JSONObject toJson(SootField v) {
		JSONObject ob = new JSONObject();
		String name = Scene.v().quotedNameOf(v.getName());
		String type = v.getType().toString();
		String modifier = Modifier.toString(v.getModifiers());
		ob.put("Jimple", "SootField");
		ob.put("type", type);
		ob.put("name", name);
		ob.put("modifier", modifier);
		return ob;
	}
}
