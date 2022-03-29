package jimple.graph;

public interface AbstractNode {
	public abstract NodeType getType();
	public abstract String getTypeName() ;
	public abstract String toString() ;
	public abstract int hashCode() ;
	public abstract int getID() ;

}
