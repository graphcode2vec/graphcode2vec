package jimple.io;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import org.jgrapht.graph.AbstractBaseGraph;
import org.jgrapht.nio.DefaultAttribute;
import org.jgrapht.nio.Attribute;
import org.jgrapht.nio.dot.DOTExporter;
import org.json.JSONArray;
import org.json.JSONObject;

import java_cup.version;
import jimple.graph.AbstractNode;
import jimple.graph.Edge;
import jimple.graph.Node;

public class ExportGraphDot {
	   private Path outputfolder;
       public ExportGraphDot(String p, boolean isCreate) {
    	   this.outputfolder = Paths.get(p);
    	   if(!Files.isDirectory(this.outputfolder) && isCreate) {
    		   try {
				Files.createDirectories(this.outputfolder);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	   }
       }
       
       
       public void writeGraph( AbstractBaseGraph<AbstractNode, Edge> graph, File file) {
    	   DOTExporter<AbstractNode, Edge> exporter 
    	   = new DOTExporter<AbstractNode, Edge>( v -> Integer.toString( v.getID()) );
    	   exporter.setVertexAttributeProvider( (v) -> {
    		   Map<String, Attribute> map = new LinkedHashMap<>();
               //map.put("label", DefaultAttribute.createAttribute(v.toString()));
               map.put("type", DefaultAttribute.createAttribute( v.getTypeName() ) );
               return map;
    	   } );
    	   exporter.setEdgeAttributeProvider((e) -> {
    		   Map<String, Attribute> map = new LinkedHashMap<>();
    		   map.put("label", DefaultAttribute.createAttribute(e.getTypeName()));
               return map;
    	   });
    	   exporter.exportGraph(graph, file);
       }
       
     
       public void writeJSON(JSONObject json, File file) {
	   		try {
	   			FileWriter filewriter = new FileWriter(file);
	   			filewriter.write(json.toString());
	   			filewriter.close();
	   		} catch ( IOException e) {
	   			// TODO Auto-generated catch block
	   			e.printStackTrace();
	   		}
       }
       
       public String writeGraph( AbstractBaseGraph<AbstractNode, Edge> graph) {
    	   Writer w1 = new StringWriter();
    	   DOTExporter<AbstractNode, Edge> exporter 
    	   = new DOTExporter<AbstractNode, Edge>( v -> Integer.toString( v.getID()) );
    	   exporter.setVertexAttributeProvider( (v) -> {
    		   Map<String, Attribute> map = new LinkedHashMap<>();
               //map.put("label", DefaultAttribute.createAttribute(v.toString()));
               map.put("type", DefaultAttribute.createAttribute( v.getTypeName() ) );
               return map;
    	   } );
    	   exporter.setEdgeAttributeProvider((e) -> {
    		   Map<String, Attribute> map = new LinkedHashMap<>();
    		   map.put("label", DefaultAttribute.createAttribute(e.getTypeName()));
               return map;
    	   });
    	   exporter.exportGraph(graph, w1);
    	   String dotGraph = w1.toString();
    	   return dotGraph;
       }
       
}
