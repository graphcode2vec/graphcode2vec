package jimple.intrapdg;
import java.util.Iterator;
import java.util.Map;
import soot.SceneTransformer;
import soot.Body;
import soot.MethodOrMethodContext;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Transform;
import soot.jimple.toolkits.callgraph.CHATransformer;
import soot.jimple.toolkits.callgraph.CallGraph;
import soot.jimple.toolkits.callgraph.Targets;
import soot.jimple.toolkits.callgraph.Edge;
import soot.Context;
public class CallgraphAnalysis extends SceneTransformer{

	@Override
	protected void internalTransform(String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
		 CHATransformer.v().transform();
		 CallGraph cg = Scene.v().getCallGraph();
		 
		 System.out.print(cg.size());
	}

}
