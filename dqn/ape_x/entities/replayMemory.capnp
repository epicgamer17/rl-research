@0x95bed824078b511a;

struct TransitionBatch {
  # A batch of transitions to be pushed to the replay memory

  ids @0 :List(Text);                # uuid of the transition, generated by actors
  observations @1 :Data;             # blob - picked list of numpy multiarrays
  nextObservations @2 :Data;         # blob - picked list of numpy multiarrays
  actions @3 :List(UInt8);           # discrete action of actor
  rewards @4 :List(Float32);         # reward of actor
  dones @5 :List(Bool);              # whether the episode is done
  priorities @6 :List(Float32) = [ ];      # initial estimate of the priority of the transition, generated by actors
  indices @7 :List(Int32) = [ ];           # indices of the transitions in the replay memory
  weights @8 :List(Float32) = [ ];         # weights
}

struct PriorityUpdate {
  ids @0 :List(Text);                # uuid of the transition, generated by actors
  indices @1 :List(Int32);           # indices of the transitions in the replay memory
  losses @2 :List(Float32);          # losses of the transitions
}