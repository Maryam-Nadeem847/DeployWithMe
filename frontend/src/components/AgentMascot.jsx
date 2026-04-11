import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import robotImg from "../assets/robot.png";

const STATES = ["Reasoning...", "Planning...", "Building...", "Executing..."];

export default function AgentMascot() {
  const [idx, setIdx] = useState(0);
  const stateText = useMemo(() => STATES[idx % STATES.length], [idx]);

  useEffect(() => {
    const id = setInterval(() => setIdx((v) => v + 1), 2500);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex w-full flex-col items-center">
      <div className="flex w-full items-center justify-center">
        <div className="robot-mascot-float flex items-center justify-center">
          <div className="robot-mascot-breathe flex items-center justify-center">
            <div
              className="robot-mascot-glow-pulse inline-flex items-center justify-center rounded-full bg-white"
              style={{ mixBlendMode: "multiply" }}
            >
              <img
                src={robotImg}
                alt="Deploy agent mascot"
                className="h-72 w-72 object-contain md:h-96 md:w-96"
              />
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 min-h-6">
        <AnimatePresence mode="wait">
          <motion.p
            key={stateText}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.35 }}
            className="bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text font-mono text-sm tracking-widest text-transparent"
          >
            {stateText}
          </motion.p>
        </AnimatePresence>
      </div>
    </div>
  );
}
