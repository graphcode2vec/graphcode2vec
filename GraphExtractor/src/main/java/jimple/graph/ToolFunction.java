package jimple.graph;
import static jimple.graph.ToolFunction.DEV_MODE;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.jar.*;

import org.jgrapht.graph.AbstractBaseGraph;
import org.json.JSONObject;
import org.json.JSONTokener;

import soot.Local;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.UnitGraph;
import soot.toolkits.graph.pdg.IRegion;
import soot.toolkits.graph.pdg.PDGNode;
import soot.toolkits.graph.pdg.PDGRegion;
import soot.toolkits.graph.pdg.Region;
import soot.toolkits.scalar.SimpleLocalDefs;
import soot.util.Chain;
public class ToolFunction {
	public static final boolean DEV_MODE = true;
	
	public static void extractJars(String jarpath, String destDir) {
		try {
			JarFile jar = new JarFile(jarpath);
			Enumeration enumEntries = jar.entries();
			
			while(enumEntries.hasMoreElements()) {
				JarEntry file = (JarEntry) enumEntries.nextElement();
				java.io.File f = new java.io.File(destDir + java.io.File.separator + file.getName());
			    if (file.isDirectory()) { // if its a directory, create it
			        f.mkdir();
			        continue;
			    }
			    java.io.InputStream is = jar.getInputStream(file); // get the input stream
			    java.io.FileOutputStream fos = new java.io.FileOutputStream(f);
			    while (is.available() > 0) {  // write contents of 'is' to 'fos'
			        fos.write(is.read());
			    }
			    fos.close();
			    is.close();
			}
			jar.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static HashMap<Unit, AbstractNode> generateDefID(UnitGraph graph, AtomicInteger id_counter) {
		Iterator<Unit> itr = graph.iterator();
		HashMap<Unit, AbstractNode> idList =  new HashMap<Unit, AbstractNode>();
		while (itr.hasNext() ) {
			Unit u = itr.next();
			if(u==null) {
				System.out.println("Null");
			}
			idList.put(u, new Node(u, id_counter.get()));
			id_counter.incrementAndGet();
		}
		return idList;
	}
	
	public static HashMap<Local, AbstractNode> generateLocalID(UnitGraph graph, AtomicInteger id_counter) {
		Chain<Local> local_chains = graph.getBody().getLocals();
		Iterator<Local> itr = local_chains.iterator();
		HashMap<Local, AbstractNode> idList =  new HashMap<Local, AbstractNode>();
		while (itr.hasNext() ) {
			Local u = itr.next();
			idList.put(u, new LocalNode(u, id_counter.get()));
			id_counter.incrementAndGet();
		}
		return idList;
	}
	
	public static void addContrloFlowNodes(UnitGraph graph, AbstractBaseGraph<AbstractNode, Edge> pdggraph,
			AbstractBaseGraph<AbstractNode, Edge> cfggraph, HashMap<Unit, AbstractNode> idList) {
		for(Unit u : graph) {
			List<Unit> succsUnits = graph.getSuccsOf(u);
			AbstractNode n_u = idList.get(u);
			pdggraph.addVertex( n_u );
			cfggraph.addVertex( n_u );

			for(Unit t: succsUnits) {
				AbstractNode n_t = idList.get(t);
				pdggraph.addVertex(n_t);
				cfggraph.addVertex(n_t);
				cfggraph.addEdge( n_u, n_t,
						new Edge(n_u, n_t, 
						EdgeType.Controlfow));
			}
		}
	}
	
	public static void addLocals(UnitGraph graph, AbstractBaseGraph<AbstractNode, Edge> cfg,
			AbstractBaseGraph<AbstractNode, Edge> pdg,
			HashMap<Unit, AbstractNode> idList, HashMap<Local, AbstractNode> localIdlist) {
		// add local to graph
		Chain<Local> localchains = graph.getBody().getLocals();
		SimpleLocalDefs df = new SimpleLocalDefs(graph);
		for(Local local: localchains) {
			AbstractNode t1 = localIdlist.get(local);
			//System.out.println(local.toString());
			pdg.addVertex(t1);
			cfg.addVertex(t1);
			List<Unit> def_local = df.getDefsOf(local);
			for(Unit su: def_local) {
				AbstractNode s1 = idList.get(su);
				pdg.addEdge(s1, t1, new Edge(s1, t1, EdgeType.VariableDeclaration));
				cfg.addEdge(s1, t1, new Edge(s1, t1, EdgeType.VariableDeclaration));
			}
		}
	}
	
	public static void addDefUseEdges(UnitGraph graph, AbstractBaseGraph<AbstractNode, Edge> resgraph,
			HashMap<Unit, AbstractNode> idList) {
		SimpleLocalDefs df = new SimpleLocalDefs(graph);
		
	    for(Unit u: graph) {
	    	//System.out.println(u.toString());
	    	List<ValueBox> box = u.getUseBoxes(); // find the use value box of the Unit

	    	for(ValueBox vb: box) {   
	    		Value v = vb.getValue();
	    		if( v instanceof Local) {
	    			List<Unit> dfsites = df.getDefsOfAt((Local)v, u);
	    			for(Unit t: dfsites) {
	    				resgraph.addEdge(idList.get(u), idList.get(t), new Edge(idList.get(u),
	    						idList.get(t), EdgeType.DataDependence)); //add the edge
	    			}
	    		}
	    	}
	    }
		
	}
	
	/**
	 * 1. Visit PDG graph from the beginning PDGNode
	 * 2. If the cur node has the control dependence with its successor
	 * 	  add edges between Units of the curt PDGNode and Units of its successor
	 * **/
	public static void addControlDependenceEdges(UnitGraph graph, 
			AbstractBaseGraph<AbstractNode, Edge> resgraph,
			HashMap<Unit, AbstractNode> idList) {
			// Build Program Region Graph
		    // Notice: The graph also have controlflow edge
		 HashMutablePDGVariant pdg = new HashMutablePDGVariant(graph);
		 HashMap<Integer, PDGNode> processed = new HashMap<Integer, PDGNode>();
		 HashMap<Unit, Integer> record =new HashMap<Unit, Integer>();
		 Queue<PDGNode> nodes = new LinkedList<PDGNode>();
		 List<PDGRegion> rlist = pdg.getPDGRegions();
		 nodes.offer(pdg.GetStartNode());

		 while (nodes.peek() != null) {
		      PDGNode node = (PDGNode) nodes.remove();
		      //processed.put(node.setVisited(v);, value)
		      node.setVisited(true);
		      List<PDGNode> succs = pdg.getSuccsOf(node);
		      for (PDGNode succ : succs) {
		    	  //System.out.println(succ.getVisited());
		    	  if(succ.getVisited()) {
		    		  continue;
		    	  }
		    	  if(nodes.contains(succ)) {
		    		  continue;
		    	  }
		    	  nodes.offer(succ);
		      }
		      
		      if(node.getAttrib() == PDGNode.Attribute.NORMAL) {
		      //pdgnodeCFGEDGE( node, graph, resgraph, idList, record );
				  continue;
		      }
		      List<PDGNode>  dependNodes = pdg.getDependents(node);
		      //System.out.println("==============================");
		      //System.out.println(node.toString());
		      List<Unit> dunits = getAllUnits(node);
		      //for(Unit u: dunits) {
		    //	  System.out.println( u.toString());
		     // }
		      for(PDGNode pdgnode: dependNodes) {
		    	  List<String> tmp = pdg.getLabelsForEdges(node, pdgnode);
		    	  //System.out.println("==============================");
			      //System.out.println(pdgnode.toString());
		    	  //for(String s: tmp) {
		    	//	  System.out.println(s);
		    	 // }
		    	  List<Unit> us = getAllUnits(pdgnode);
		    	//  for(Unit u: us) {
			    //	  System.out.println( u.toString());
			    //  }
		    	  //pdgnodeCFGEDGE( pdgnode, graph, resgraph, idList, record );
		    	  for(Unit uu: dunits) {
		    		  for(Unit u: us) {
			    		  resgraph.addEdge(idList.get(u), idList.get(uu), new Edge(idList.get(u), 
			            			 idList.get(uu), EdgeType.ControlDependence));
			    		  record.put(u, 1);
			    		  record.put(uu, 1);
			    	  }
		    	  }
		      }
		      
		    }
		 
		 Iterator<Unit> allunits = graph.iterator();
		 while(allunits.hasNext()) {
			 Unit uuu = allunits.next();
			 if(!record.containsKey(uuu)) {
				 List<Unit> uss = graph.getSuccsOf(uuu);
				 if(resgraph.degreeOf(idList.get(uuu)) !=0 )
					 continue;
				  for(Unit u: uss) {
					  resgraph.addEdge(idList.get(uuu), idList.get(u), new Edge(idList.get(uuu), 
		            			 idList.get(u), EdgeType.Controlfow)); 
				  }
				  List<Unit> upos = graph.getPredsOf(uuu);
				  for(Unit up: upos) {
					  resgraph.addEdge(idList.get(up), idList.get(uuu), new Edge(idList.get(up), 
		            			 idList.get(uuu), EdgeType.Controlfow)); ;
				  }
			 }
		 }
		 pdg.clearAll();
		 pdg = null;
		 processed = null;
		 record = null;
		 nodes = null;
		 rlist = null;
		 }
	
	/**
	  * Fetch all the Units in PDGNode
	 * @throws Exception 
	  * **/
	  public static List<Unit> getAllUnits(PDGNode node){
		  List<Unit> res = new ArrayList<Unit>();
		  Object obj_node = node.getNode();
		  if( obj_node instanceof IRegion) {
			  return ((IRegion)obj_node).getUnits();
		  }else if( obj_node instanceof Region ) {
			  return ((Region)obj_node).getUnits();
		  }else if( obj_node instanceof PDGRegion) {
			  return ((PDGRegion)obj_node).getUnits();
		  }else if(obj_node instanceof Block) {
			  Iterator<Unit> itrs = ((Block)(node.getNode())).iterator(); 
			  while( itrs.hasNext() ) {
				  res.add(itrs.next());
			  }
		  }else {
			 // throw new Exception("PDGnode contain unknow object");
			  System.out.println("Warning getAllUnits in IntraAnalysis, PDGNode contains unknown objects");
		  }
		  return res;
	  }
	  
	  /**
		  * Fetch all the Units in PDGNode
		 * @throws Exception 
		  * **/
	  public static List<Block> getAllBlocks(PDGNode node){
			  List<Block> res = new ArrayList<Block>();
			  Object obj_node = node.getNode();
			  if( obj_node instanceof IRegion) {
				  return ((IRegion)obj_node).getBlocks();
			  }else if( obj_node instanceof Region ) {
				  return ((Region)obj_node).getBlocks();
			  }else if( obj_node instanceof PDGRegion) {
				  return ((PDGRegion)obj_node).getBlocks();
			  }else if(obj_node instanceof Block) {
				  Block h = ((Block)(node.getNode())); 
				  res.add(h);
			  }else {
				 // throw new Exception("PDGnode contain unknow object");
				  System.out.println("Warning getAllUnits in IntraAnalysis, PDGNode contains unknown objects");
			  }
			  return res;
		  }
	  
	  public static Path createPathByClassName(String classname, String outputfolder) {
			//List<String> list = new ArrayList<String>(Arrays.asList(classname.split(".")));
			String classpath = classname.replace(".", File.separator);
			Path path = Paths.get(outputfolder, classpath);
			try {
				Files.createDirectories(path);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return path;
		}
		 
	  public static ArrayList<String> storeUnitStatement(UnitGraph graph) {
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
		
	  public static JSONObject readJson(String jsonfile, String classname) throws FileNotFoundException {
		  File initialFile = new File(jsonfile);
		  InputStream is = new FileInputStream(initialFile);
	      JSONTokener tokener = new JSONTokener(is);
	      JSONObject object = new JSONObject(tokener);
	      JSONObject classmethod = object.getJSONObject(classname);
	      JSONObject newclassmethod = new JSONObject();
	      Iterator<String> keys = classmethod.keys();
	      while(keys.hasNext()) {
	    	  String k = keys.next();
	    	  String v = classmethod.getString(k);
	    	  newclassmethod.put(v, k);
	      }
		return newclassmethod;
	  }
}
