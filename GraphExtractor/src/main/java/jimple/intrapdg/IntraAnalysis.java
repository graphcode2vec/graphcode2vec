package jimple.intrapdg;

import static jimple.graph.ToolFunction.DEV_MODE;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.jgrapht.graph.AbstractBaseGraph;
import org.jgrapht.graph.DirectedPseudograph;
import org.json.JSONArray;
import org.json.JSONObject;

import jimple.graph.AbstractNode;
import jimple.graph.Edge;
import jimple.graph.EdgeType;
import jimple.graph.Entry;
import jimple.graph.Exist;
import jimple.graph.HashMutablePDGVariant;
import jimple.graph.LocalNode;
import jimple.graph.Node;
import jimple.graph.NodeType;
import jimple.graph.ToolFunction;
import jimple.instruction.encoding.JimpleStmt;
import jimple.io.Common;
import jimple.io.ExportGraphDot;
import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.Modifier;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Type;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.InvokeStmt;
import soot.jimple.toolkits.callgraph.CHATransformer;
import soot.jimple.toolkits.callgraph.CallGraph;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.ExceptionalUnitGraph;
import soot.toolkits.graph.UnitGraph;
//import soot.toolkits.graph.pdg.HashMutablePDG;
import soot.toolkits.graph.pdg.IRegion;
import soot.toolkits.graph.pdg.PDGNode;
import soot.toolkits.graph.pdg.PDGRegion;
import soot.toolkits.graph.pdg.Region;
import soot.toolkits.scalar.SimpleLocalDefs;
import soot.util.Chain;

public class IntraAnalysis  extends BodyTransformer{
	public static int total = 0;
	public static int failed = 1;
	private ExportGraphDot exporter;
	private String outputfolder ;
	private static JSONObject class_methods = new JSONObject();
	private static long method_counter = 0;

    private static final Object lock = new Object();
    //private JimpleStmt converter = new JimpleStmt();
    // method graph
    private static JSONObject methodgraph_org_cfg = new  JSONObject();
    private static JSONObject methodgraph_org_pdg = new JSONObject();
    private static JSONObject methodgraph_dv_cfg = new  JSONObject();
    private static JSONObject methodgraph_dv_pdg = new JSONObject();
    private static JSONObject methodgraph_du_cfg = new  JSONObject();
    private static JSONObject methodgraph_dvdu_cfg = new JSONObject();
    private static JSONObject statements = new JSONObject();
    private static JSONObject metainfo = new JSONObject();
    private static JSONObject raw_jimpleIns = new JSONObject();
	public IntraAnalysis(String p) {
		this.exporter = new ExportGraphDot(p, true);
		this.outputfolder = p;

	}
	

    //class_method_id_mapping
	public void writeJson(String name, JSONObject res) {	
		try {
			FileWriter filewriter = new FileWriter(Paths.get(this.outputfolder, 
					name).toFile());
			filewriter.write(res.toString());
			
			filewriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void storeResults() {
		this.writeJson("class_method_id_mapping.json", this.class_methods);
		this.writeJson("ORG_CFG.json", this.methodgraph_org_cfg);
		this.writeJson("ORG_PDG.json", this.methodgraph_org_pdg);
		this.writeJson("DU_CFG.json", this.methodgraph_du_cfg);
		this.writeJson("DV_CFG.json", this.methodgraph_dv_cfg);
		this.writeJson("DV_PDG.json", this.methodgraph_dv_pdg);
		this.writeJson("DVDU_CFG.json", this.methodgraph_dvdu_cfg);
		this.writeJson("TypeInfo.json", this.metainfo);
		this.writeJson("statement.json", this.statements);
		this.writeJson("RawIns.json", this.raw_jimpleIns);
	}
	
	public ArrayList<String> storeUnitStatement(UnitGraph graph) {
		Iterator<Unit> itr = graph.iterator();
		ArrayList<String> node_statement = new ArrayList<String>();
		while(itr.hasNext()) {
			Unit u = itr.next();
			node_statement.add(u.toString());
		}
		Chain<Local> localchains = graph.getBody().getLocals();
		for(Local l:localchains) {
			node_statement.add(l.getType().toString() +" "+l.toString());
		}
		return node_statement;
	}
	
	public void writeStatement(ArrayList<String> node_statement, File file) {
		JSONArray jsonarray = new JSONArray(node_statement);
		try {
			FileWriter filewriter = new FileWriter(file);
			filewriter.write(jsonarray.toString());
			filewriter.close();
		} catch ( IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void storeJSON(JSONObject ob, File file) {
		try {
			FileWriter filewriter = new FileWriter(file);
			filewriter.write(ob.toString());
			filewriter.close();
		} catch ( IOException e) {
			// TODO Auto-generated catch block
						e.printStackTrace();
		}
	}
	
	public JSONObject getUnitMetaInfo(HashMap<Unit, AbstractNode> info) throws Exception {
		JSONObject metainfo = new JSONObject();
		Iterator<java.util.Map.Entry<Unit, AbstractNode>> itr = info.entrySet().iterator();
		JimpleStmt converter = new JimpleStmt();
		while( itr.hasNext() ) {
			java.util.Map.Entry<Unit, AbstractNode> element = itr.next();
			Unit u = element.getKey();
			AbstractNode n = element.getValue();
			int id = n.getID();
			converter.cleanVisitedList();
			JSONObject unitmeta = converter.toJson(u);
			//System.out.println( u.toString() + " #  " +unitmeta.toString());
			metainfo.put(Integer.toString(id), unitmeta);
		}
		return metainfo;
	}
	
	public JSONObject getRawUnitIns(HashMap<Unit, AbstractNode> info) {
		JSONObject rawIns = new JSONObject();
		Iterator<java.util.Map.Entry<Unit, AbstractNode>> itr = info.entrySet().iterator();
		while( itr.hasNext() ) {
			java.util.Map.Entry<Unit, AbstractNode> element = itr.next();
			Unit u = element.getKey();
			AbstractNode n = element.getValue();
			int id = n.getID();
			String unitmeta = u.toString();
			//System.out.println( u.toString() + " #  " +unitmeta.toString());
			rawIns.put(Integer.toString(id), unitmeta);
		}
		return rawIns;
		
	}
	
	public JSONObject getRawLocalIns(HashMap<Local, AbstractNode> info) {
		JSONObject localIns = new JSONObject();
		Iterator<java.util.Map.Entry<Local, AbstractNode>> itr = info.entrySet().iterator();
		while( itr.hasNext() ) {
			java.util.Map.Entry<Local, AbstractNode> element = itr.next();
			Local u = element.getKey();
			AbstractNode n = element.getValue();
			int id = n.getID();
			String unitmeta = u.getType().toString() + " " +u.toString();
			//System.out.println( u.toString() + " #  " +unitmeta.toString());
			localIns.put(Integer.toString(id), unitmeta);
		}
		return localIns;
	}
	public JSONObject getLocalMetaInfo(HashMap<Local, AbstractNode> info) throws Exception{
		JSONObject metainfo = new JSONObject();
		Iterator<java.util.Map.Entry<Local, AbstractNode>> itr = info.entrySet().iterator();
		JimpleStmt converter = new JimpleStmt();
		while( itr.hasNext() ) {
			java.util.Map.Entry<Local, AbstractNode> element = itr.next();
			Local u = element.getKey();
			AbstractNode n = element.getValue();
			int id = n.getID();
			converter.cleanVisitedList();
			JSONObject unitmeta = converter.toJson(u);
			//System.out.println( u.toString() + " #  " +unitmeta.toString());
			metainfo.put(Integer.toString(id), unitmeta);
		}
		return metainfo;
	}
	
	public JSONObject calledMethod(HashMap<Unit, AbstractNode> unitlits) {
		CallGraph cg = Scene.v().getCallGraph();
		JSONObject ob = new JSONObject();
		for (Map.Entry<Unit, AbstractNode> entry : unitlits.entrySet()) {
			Unit u = entry.getKey();
			Iterator<soot.jimple.toolkits.callgraph.Edge> it = cg.edgesOutOf(u);
			JSONArray ta = new JSONArray();
			while(it.hasNext()) {
				soot.jimple.toolkits.callgraph.Edge e = it.next();
				SootMethod method = e.tgt();
				JSONObject tob = new JSONObject();
				if (method.isConcrete() && !method.getDeclaringClass().isLibraryClass()) {
					tob.put("signature", method.getSignature());
					tob.put("class", method.getDeclaringClass().getName());
					ta.put(tob);
				}
				
			}
			if(ta.length()!=0) {
				ob.put(Integer.toString(unitlits.get(u).getID()), ta);
			}
			
			
		}
		return ob;
		
	}
	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		//int startline = Integer.MAX_VALUE;
		int endline = -1;
		int startline = b.getMethod().getJavaSourceStartLineNumber();
		for(Unit u: b.getUnits()) {
			int lno = u.getJavaSourceStartLineNumber();
			if(lno==-1)
				continue;
			endline = lno > endline? lno : endline;
			startline = lno > startline? startline : lno;
		}
		
		
		if(!b.getMethod().isConcrete()) {
			return;
		}
		boolean constructor = false;
		constructor = b.getMethod().isConstructor();
		
		//System.out.println(phaseName);
		AtomicInteger id_counter = new AtomicInteger(0);  // assign unique id to Unit
		UnitGraph graph = new ExceptionalUnitGraph(b);
		ArrayList<String> nodestatement = this.storeUnitStatement(graph);
		HashMap<Unit, AbstractNode> idList = ToolFunction.generateDefID(graph, id_counter);
		JSONObject calledmethods = calledMethod(idList);
		HashMap<Local, AbstractNode> localIdlist = ToolFunction.generateLocalID(graph, id_counter);
		
		Edge e = new Edge(); // create CFG and PDG Graph
		//System.out.println("Graph");
		AbstractBaseGraph<AbstractNode, Edge> intragraph = new DirectedPseudograph<AbstractNode, 
				Edge>( e.getClass() );
		AbstractBaseGraph<AbstractNode, Edge> cfggraph = new DirectedPseudograph<AbstractNode, 
				Edge>(  e.getClass()); 
		
		//add control flow edges
		ToolFunction.addContrloFlowNodes(graph, intragraph, cfggraph, idList);
		//add data dependence edges, provided by Soot
		ToolFunction.addDefUseEdges(graph, intragraph, idList);
		//add control dependence edges
		ToolFunction.addControlDependenceEdges(graph, intragraph, idList);
		
		String signature = b.getMethod().getDeclaration();
		String className = b.getMethod().getDeclaringClass().getName();
		JimpleStmt converter = new JimpleStmt();
		JSONObject methodinfo = converter.toJson(b.getMethod().makeRef());
		if(constructor) {
			String classnameOfmethod = methodinfo.getString("class");
			String[] res = classnameOfmethod.split("\\$");
			classnameOfmethod= res[res.length-1];
			String normalizedMethodName = Common.normalizeName(classnameOfmethod, Common.BlankWord);
			ArrayList<String> splitNameParts = Common.splitToSubtokens(classnameOfmethod);
			String splitName = normalizedMethodName;
			if (splitNameParts.size() > 0) {
				splitName = splitNameParts.stream().collect(Collectors.joining(Common.internalSeparator));
			}
			methodinfo.put("normalized_name", splitName);
		}
		methodinfo.put("calledmethod", calledmethods);
		methodinfo.put("start", startline);
		methodinfo.put("end", endline);
		//Path p = this.createPathByClassName(className);
		
		String name;
		synchronized (this.lock) {   // require lock thread-safe to assgin method id
			if(!this.class_methods.has(className)) {
				this.class_methods.put(className, new JSONObject());
			}
			//System.out.println(className);
			this.class_methods.getJSONObject(className).put(Long.toString(this.method_counter), methodinfo);
			name = Long.toString(this.method_counter);
			this.method_counter++;	
		}
		try {
			JSONObject localmeta = this.getLocalMetaInfo(localIdlist);
			JSONObject unitmeta = this.getUnitMetaInfo(idList);
			JSONObject metainfo = new JSONObject();
			metainfo.put("Unit", unitmeta);
			metainfo.put("Local", localmeta);
			synchronized (this.lock) {  
				this.metainfo.put(name, metainfo);
			}
			//this.storeJSON(metainfo, Paths.get(p.toString(), "meta_Stmt"+name+".json").toFile());
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			//System.exit(failed);
		}	
		try {	
			JSONObject localIns = this.getRawLocalIns(localIdlist);
			JSONObject unitIns = this.getRawUnitIns(idList);
			JSONObject rawins = new JSONObject();
			rawins.put("Unit", unitIns);
			rawins.put("Local", localIns);
			synchronized (this.lock) {  
				this.raw_jimpleIns.put(name, rawins);
			}
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			//System.exit(failed);
		}
		
		// original graphs
		//this.exporter.writeGraph(cfggraph, Paths.get(p.toString(), "ORG_CFG_"+name).toFile());
		//this.exporter.writeGraph(intragraph, Paths.get(p.toString(), "ORG_PDG_"+name).toFile());
		String org_cfg = this.exporter.writeGraph(cfggraph);
		String org_pdg = this.exporter.writeGraph(intragraph);
		synchronized (this.lock) {
			this.methodgraph_org_cfg.put(name, org_cfg);
			this.methodgraph_org_pdg.put(name, org_pdg);
		}
		// add locals to the graphs
		ToolFunction.addLocals(graph, cfggraph, intragraph, idList, localIdlist);
		//this.exporter.writeGraph(cfggraph, Paths.get(p.toString(), "DV_CFG_"+name).toFile());
		//this.exporter.writeGraph(intragraph, Paths.get(p.toString(), "DV_PDG_"+name).toFile());
		String dv_cfg = this.exporter.writeGraph(cfggraph);
		String dv_pdg = this.exporter.writeGraph(intragraph);
		synchronized (this.lock) {
			this.methodgraph_dv_cfg.put(name, dv_cfg);
			this.methodgraph_dv_pdg.put(name, dv_pdg);
		}	
		//add def-use edge to cfg graphs
		AbstractBaseGraph<AbstractNode, Edge> cfggraph_copy = new DirectedPseudograph<AbstractNode, 
				Edge>(Edge.class);
		AbstractBaseGraph<AbstractNode, Edge> intragraph_copy = new DirectedPseudograph<AbstractNode, 
				Edge>(Edge.class);
		
		ToolFunction.addContrloFlowNodes(graph, intragraph_copy, cfggraph_copy, idList);
		ToolFunction.addDefUseEdges(graph, cfggraph_copy, idList);
		//this.exporter.writeGraph(cfggraph_copy, Paths.get(p.toString(), "DU_CFG_"+name).toFile());
		String du_cfg = this.exporter.writeGraph(cfggraph_copy);
		
		synchronized (this.lock) {
			this.methodgraph_du_cfg.put(name, du_cfg);
		}
		// add locals to the graphs with Def-Use CFG
		ToolFunction.addLocals(graph, cfggraph_copy, intragraph_copy, idList, localIdlist);
		//this.exporter.writeGraph(cfggraph_copy, Paths.get(p.toString(), "DV_DU_CFG_"+name).toFile());
		String dv_du_cfg = this.exporter.writeGraph(cfggraph_copy);
		//this.writeStatement(nodestatement,  Paths.get(p.toString(), "statement_"+name+".json").toFile());
		JSONArray jsonarray = new JSONArray(nodestatement);
		
		synchronized (this.lock) {  
			this.statements.put(name, jsonarray);
			this.methodgraph_dvdu_cfg.put(name, dv_du_cfg);
		}
		graph = null;
		cfggraph = null;
		intragraph = null;
		cfggraph_copy = null;
		intragraph_copy = null;
		converter = null;
		//System.out.println("Finish");
	}
	

	public Path createPathByClassName(String classname) {
		//List<String> list = new ArrayList<String>(Arrays.asList(classname.split(".")));
		String classpath = classname.replace(".", File.separator);
		Path path = Paths.get(this.outputfolder, classpath);
		try {
			Files.createDirectories(path);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return path;
	}

}