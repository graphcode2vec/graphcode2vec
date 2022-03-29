package jimple;
import static jimple.graph.ToolFunction.DEV_MODE;

import java.io.File;

import soot.Body;
import soot.BodyTransformer;
import soot.G;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
//import soot.SootMethod;
import soot.Transform;
//import soot.Unit;
import soot.options.*;
//import soot.toolkits.graph.ExceptionalUnitGraph;
//import soot.toolkits.graph.pdg.HashMutablePDG;
import soot.util.HashChain;
import jimple.cmd.MyOptions;
//import soot.util.cfgcmd.CFGToDotGraph;
//import soot.util.dot.DotGraph;
import  jimple.intrapdg.*;
import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
public class IntraProcedureAnalysis {
	static Logger log = Logger.getLogger(IntraProcedureAnalysis.class);
	CommandLine cmd;
	public void setup(String jarpath, String storepath, BodyTransformer intradd) {
		//System.out.println("setup");
		// Analysis Jar files
		G.reset();
		String classesDir = jarpath;
	    // set classpath
	    String jreDir = System.getProperty("java.home") + File.separator
			    		+ "lib"+ File.separator +"jce.jar";
	    String jceDir = System.getProperty("java.home") + File.separator 
			    		+ "lib" + File.separator +"rt.jar";
	    String path = jreDir + File.pathSeparator + jceDir + File.pathSeparator + classesDir;
	    //System.out.println(path);
			    
	    //add an intra-procedural analysis phase to Soot
	    Options.v().set_prepend_classpath(false);
	    //Options.v().set_soot_modulepath(sootClassPath());
		PackManager.v().getPack("jtp").add(new Transform("jtp.IntraDDEDGES", intradd));
		//PackManager.v().getPack("wjtp").add(new Transform("wjtp.myTrans", new CallgraphAnalysis()));
//		PackManager.v().getPack("jb").remove("jb.tr");
//		PackManager.v().getPack("jb").add(new Transform("jb.tr", new TypeAssigner_Custom() ));
	    Options.v().set_process_dir(Arrays.asList(classesDir.split(File.pathSeparator)));

	    Scene.v().setSootClassPath(path);
	    Options.v().set_keep_line_number(true);
	    Options.v().set_whole_program(true);
		Options.v().set_allow_phantom_refs(true);
		Options.v().set_verbose(false);
		//System.out.println("loadNecessaryClasses");
		Scene.v().addBasicClass("io.netty.channel.ChannelFutureListener",  SootClass.SIGNATURES);
		Scene.v().addBasicClass("io.netty.channel.ChannelFutureListener",  SootClass.HIERARCHY);
		Scene.v().addBasicClass("org.hibernate.test.c3p0.JdbcCompatibilityTest",SootClass.SIGNATURES);
		Scene.v().addBasicClass("org.hibernate.test.bytecode.enhancement.otherentityentrycontext.OtherEntityEntryContextTest"
				,SootClass.SIGNATURES);
		try {
			if(this.cmd.hasOption("m")) {
				SootClass c = Scene.v().forceResolve(this.cmd.getOptionValue("m"), SootClass.BODIES);
				c.setApplicationClass();
				Scene.v().loadNecessaryClasses();
				SootMethod method = c.getMethodByName("main");
				List entryPoints = new ArrayList();
				entryPoints.add(method);
				Scene.v().setEntryPoints(entryPoints);
			}else {
				Scene.v().loadNecessaryClasses();
			}
		}catch(Exception e){
			SootClass c = Scene.v().forceResolve("deadlockExamples."+this.cmd.getOptionValue("m"), SootClass.BODIES);
			c.setApplicationClass();
			Scene.v().loadNecessaryClasses();
			SootMethod method = c.getMethodByName("main");
			List entryPoints = new ArrayList();
			entryPoints.add(method);
			Scene.v().setEntryPoints(entryPoints);
			Scene.v().loadNecessaryClasses();
		}
	
		//System.out.println("=====================");
	   // printJarClasses();
//	    SootClass m = Scene.v().getSootClass("org.apache.commons.lang3.math.NumberUtilsTest");
//	    for(SootMethod sm: m.getMethods()) {
//	    	System.out.println(sm.toString());
//	    }
		//Scene.v().setMainClass(m);
		//ArrayList<SootMethod> list=new ArrayList<SootMethod> ();
		//Scene.v().getMethod("public static void main(String[] args)");
		//list.add(m.getMethodByName("TestLang747"));
		//Scene.v().getMethod("TestLang747");
		//Scene.v().setEntryPoints(list);

	    
	}
	
	public void printJarClasses() {
		HashChain<SootClass> applicationclasses=(HashChain<SootClass>) Scene.v().getApplicationClasses();
	    for(SootClass sc: applicationclasses) {
	    	System.out.println(sc.getName()); //print class name
	    }
	}
	
	
	public void runExtractGraph(String data_source, String data_ouput, BodyTransformer intradd) {
		try {
		    this.setup(data_source, data_ouput, intradd);
		   // printJarClasses();
		    PackManager.v().runPacks();
		   // System.out.printf("PDG without any issue %d, Failed number %d \n", IntraAnalysis.total, IntraAnalysis.failed);
		  
		}catch(Exception ex) {
			ex.printStackTrace();
			System.exit(1);
		}
	}
	
	public static void main(String[] args){
		BasicConfigurator.configure();
		//log.setLevel(Level.ERROR);
		// create Options object
		MyOptions options = new MyOptions();
		// Options Extracting Graph
		options.addOption(new Option("g","graph" ,false, "Extract graph."));
        options.addOption(new Option("i", "input",true, "Input class folder."));
        options.addOption(new Option("o", "output",true, "Output result folder."));
        options.addOption(new Option("o", "output",true, "Output result folder."));
        options.addOption(new Option("j", "json", true, "Class Info JSON"));
        // Options Coverage Information
        options.addOption(new Option("m", "main",true, "Main Class."));
        options.addOption(new Option("c", "coverage",true, "Coverage Dir"));
        options.addOption(new Option("s", "source",true, "Source code folder."));
        options.addOption(new Option("p", "patchfile",true, "Patch File Paths"));
        options.addOption(new Option("h", "help",false, "Help Info"));
        // Options Mutants
        options.addOption(new Option("k", "mutant", false, "Mutation"));
        
        CommandLineParser parser = new DefaultParser();
        
        try {
			CommandLine cmd = parser.parse( options, args);
			if(cmd.hasOption("h")) {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp( "extracterGraph", options );
				return;
			}
			
		
			if(cmd.hasOption("g")) {
				if(cmd.getOptionValue("i") != null && cmd.getOptionValue("o") != null){
					IntraProcedureAnalysis intraAnalysis = new IntraProcedureAnalysis();
					intraAnalysis.cmd = cmd;
					//System.out.println(cmd.getOptionValue("i"));
					//System.out.println(cmd.getOptionValue("o"));
					IntraAnalysis intradd = new IntraAnalysis(cmd.getOptionValue("o"));
					intraAnalysis.runExtractGraph(cmd.getOptionValue("i"), cmd.getOptionValue("o"), intradd);
					intradd.storeResults();
					
				}
				
			}
			
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			System.out.println("=========================================");
			e.printStackTrace();
			System.exit(1);
		}
	    
	    //intraAnalysis.printJarClasses();
	   // intraAnalysis.methodDependGraph("./sportbugsgraph");
	}
}
