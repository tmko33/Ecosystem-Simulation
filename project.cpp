#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <omp.h>
#include <unistd.h> // Para usleep

using namespace std;

enum CreatureKind { NONE = 0, STONE = 1, BUNNY = 2, PREDATOR = 3 };

struct Position {
    int row, col;
    
    Position(int r = 0, int c = 0) : row(r), col(c) {}
    
    bool operator<(const Position& other) const {
        return row < other.row || (row == other.row && col < other.col);
    }
    
    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }
};

class Creature {
public:
    CreatureKind kind;
    Position pos;
    int generation_born;
    int last_meal_time;
    
    Creature(CreatureKind k, Position p, int born = 0, int meal = 0) 
        : kind(k), pos(p), generation_born(born), last_meal_time(meal) {}
    
    virtual ~Creature() = default;
    
    bool canReproduce(int current_gen, int reproduction_interval) const {
        return (current_gen - generation_born) >= reproduction_interval;
    }
    
    bool isStarving(int current_gen, int starvation_limit) const {
        return kind == PREDATOR && (current_gen - last_meal_time) + 1 >= starvation_limit;
    }
};

class EcosystemSimulator {
private:
    int bunny_reproduction_period;
    int predator_reproduction_period;
    int predator_starvation_limit;
    int total_generations;
    int grid_rows, grid_cols;
    int initial_population;
    
    map<Position, shared_ptr<Creature>> world_grid;
    vector<shared_ptr<Creature>> stones;
    vector<shared_ptr<Creature>> bunnies;
    vector<shared_ptr<Creature>> predators;
    
    const array<pair<int, int>, 4> movement_vectors = {{{-1,0}, {0,1}, {1,0}, {0,-1}}};
    
public:
    EcosystemSimulator() = default;
    
    void initialize(int br, int pr, int ps, int gen, int rows, int cols, int pop) {
        bunny_reproduction_period = br;
        predator_reproduction_period = pr;
        predator_starvation_limit = ps;
        total_generations = gen;
        grid_rows = rows;
        grid_cols = cols;
        initial_population = pop;
    }
    
    void addCreature(CreatureKind type, int row, int col, int birth_gen = 0, int meal_time = 0) {
        auto creature = make_shared<Creature>(type, Position(row, col), birth_gen, meal_time);
        world_grid[Position(row, col)] = creature;
        
        switch(type) {
            case STONE:
                stones.push_back(creature);
                break;
            case BUNNY:
                bunnies.push_back(creature);
                break;
            case PREDATOR:
                predators.push_back(creature);
                break;
            default:
                break;
        }
    }
    
    shared_ptr<Creature> getCreatureAt(Position pos) {
        auto iterator = world_grid.find(pos);
        return (iterator != world_grid.end()) ? iterator->second : nullptr;
    }
    
    vector<Position> findValidDestinations(shared_ptr<Creature> creature, bool hunting_mode) {
        vector<Position> destinations;
        
        for (const auto& direction : movement_vectors) {
            int new_row = creature->pos.row + direction.first;
            int new_col = creature->pos.col + direction.second;
            
            if (new_row >= 0 && new_row < grid_rows && new_col >= 0 && new_col < grid_cols) {
                Position target_pos(new_row, new_col);
                auto target_creature = getCreatureAt(target_pos);
                
                if (!hunting_mode) {
                    if (target_creature == nullptr) {
                        destinations.push_back(target_pos);
                    }
                } else {
                    if (target_creature != nullptr && target_creature->kind == BUNNY) {
                        destinations.push_back(target_pos);
                    }
                }
            }
        }
        return destinations;
    }
    
    void simulateBunnyGeneration(int current_generation) {
        map<Position, vector<shared_ptr<Creature>>> destination_groups;
        vector<map<Position, vector<shared_ptr<Creature>>>> thread_destinations(omp_get_max_threads());
        vector<shared_ptr<Creature>> current_bunnies = bunnies;
        int bunny_count = current_bunnies.size();
        bunnies.clear();
        
        #pragma omp parallel
        {
            auto& local_destinations = thread_destinations[omp_get_thread_num()];
            
            #pragma omp for schedule(dynamic, 50)
            for (int idx = 0; idx < bunny_count; idx++) {
                auto bunny = current_bunnies[idx];
                vector<Position> possible_moves = findValidDestinations(bunny, false);
                
                if (!possible_moves.empty()) {
                    int move_choice = (current_generation + bunny->pos.row + bunny->pos.col) % possible_moves.size();
                    Position new_location = possible_moves[move_choice];
                    
                    if (bunny->canReproduce(current_generation, bunny_reproduction_period)) {
                        auto offspring = make_shared<Creature>(BUNNY, bunny->pos, current_generation + 1, current_generation + 1);
                        local_destinations[bunny->pos].push_back(offspring);
                        bunny->generation_born = current_generation + 1;
                    }
                    
                    bunny->pos = new_location;
                }
                local_destinations[bunny->pos].push_back(bunny);
            }
        }
        
        #pragma omp parallel sections
        {
            #pragma section
            {
                for (auto& thread_map : thread_destinations) {
                    for (auto& [position, creature_list] : thread_map) {
                        destination_groups[position].insert(destination_groups[position].end(), 
                                                          creature_list.begin(), creature_list.end());
                    }
                }
            }
            #pragma section
            {
                world_grid.clear();
            }
        }
        
        for (auto& [position, creature_list] : destination_groups) {
            auto survivor = *max_element(creature_list.begin(), creature_list.end(),
                [current_generation](const shared_ptr<Creature>& a, const shared_ptr<Creature>& b) {
                    return (current_generation - a->generation_born) < (current_generation - b->generation_born);
                });
            
            world_grid[position] = survivor;
            bunnies.push_back(survivor);
        }
        
        #pragma omp parallel sections
        {
            #pragma section
            {
                for (auto stone : stones) world_grid[stone->pos] = stone;
                for (auto predator : predators) world_grid[predator->pos] = predator;
            }
            #pragma section
            {
                destination_groups.clear();
            }
        }
    }
    
    void simulatePredatorGeneration(int current_generation) {
        map<Position, vector<shared_ptr<Creature>>> destination_groups;
        vector<map<Position, vector<shared_ptr<Creature>>>> thread_destinations(omp_get_max_threads());
        vector<set<shared_ptr<Creature>>> thread_prey_consumed(omp_get_max_threads());
        vector<shared_ptr<Creature>> current_predators = predators;
        int predator_count = current_predators.size();
        predators.clear();
        
        #pragma omp parallel
        {
            auto& local_destinations = thread_destinations[omp_get_thread_num()];
            auto& local_consumed = thread_prey_consumed[omp_get_thread_num()];
            
            #pragma omp for schedule(dynamic, 50)
            for (int idx = 0; idx < predator_count; idx++) {
                auto predator = current_predators[idx];
                vector<Position> hunting_spots = findValidDestinations(predator, true);
                bool successfully_hunted = false;
                
                if (!hunting_spots.empty()) {
                    int hunt_choice = (current_generation + predator->pos.row + predator->pos.col) % hunting_spots.size();
                    Position prey_location = hunting_spots[hunt_choice];
                    
                    auto prey = getCreatureAt(prey_location);
                    if (prey) local_consumed.insert(prey);
                    
                    if (predator->canReproduce(current_generation, predator_reproduction_period)) {
                        auto offspring = make_shared<Creature>(PREDATOR, predator->pos, current_generation + 1, current_generation + 1);
                        local_destinations[predator->pos].push_back(offspring);
                        predator->generation_born = current_generation + 1;
                    }
                    
                    predator->pos = prey_location;
                    predator->last_meal_time = current_generation + 1;
                    local_destinations[predator->pos].push_back(predator);
                    successfully_hunted = true;
                }
                
                if (!successfully_hunted && !predator->isStarving(current_generation, predator_starvation_limit)) {
                    vector<Position> empty_spots = findValidDestinations(predator, false);
                    if (!empty_spots.empty()) {
                        int move_choice = (current_generation + predator->pos.row + predator->pos.col) % empty_spots.size();
                        Position new_location = empty_spots[move_choice];
                        
                        if (predator->canReproduce(current_generation, predator_reproduction_period)) {
                            auto offspring = make_shared<Creature>(PREDATOR, predator->pos, current_generation + 1, current_generation + 1);
                            local_destinations[predator->pos].push_back(offspring);
                            predator->generation_born = current_generation + 1;
                        }
                        predator->pos = new_location;
                    }
                    local_destinations[predator->pos].push_back(predator);
                }
            }
        }
        
        #pragma omp parallel sections
        {
            #pragma section
            {
                set<shared_ptr<Creature>> all_consumed;
                for (const auto& local_set : thread_prey_consumed) {
                    all_consumed.insert(local_set.begin(), local_set.end());
                }
                
                for (auto prey : all_consumed) {
                    auto it = find(bunnies.begin(), bunnies.end(), prey);
                    if (it != bunnies.end()) {
                        bunnies.erase(it);
                    }
                }
            }
            #pragma section
            {
                for (auto& thread_map : thread_destinations) {
                    for (auto& [position, creature_list] : thread_map) {
                        destination_groups[position].insert(destination_groups[position].end(), 
                                                          creature_list.begin(), creature_list.end());
                    }
                }
            }
            #pragma section
            {
                world_grid.clear();
            }
        }
        
        for (auto& [position, creature_list] : destination_groups) {
            sort(creature_list.begin(), creature_list.end(), 
                [](const shared_ptr<Creature>& a, const shared_ptr<Creature>& b) {
                    if (a->generation_born != b->generation_born) 
                        return a->generation_born < b->generation_born;
                    return a->last_meal_time > b->last_meal_time;
                });
            
            world_grid[position] = creature_list[0];
            predators.push_back(creature_list[0]);
        }
        
        #pragma omp parallel sections
        {
            #pragma section
            {
                for (auto stone : stones) world_grid[stone->pos] = stone;
                for (auto bunny : bunnies) world_grid[bunny->pos] = bunny;
            }
            #pragma section
            {
                destination_groups.clear();
            }
        }
    }
    
    void displayWorld(int generation) {
        // Limpiar pantalla (funciona en terminales Unix/Linux)
        cout << "\033[2J\033[1;1H";
        
        cout << "=== ECOSYSTEM SIMULATION ===" << endl;
        cout << "Generation: " << generation << " / " << total_generations << endl;
        cout << "Rabbits: " << bunnies.size() << " | Foxes: " << predators.size() << " | Rocks: " << stones.size() << endl << endl;
        
        // Leyenda
        cout << "Legend: * = Rock, R = Rabbit, F = Fox" << endl;
        cout << "Numbers show age (rabbits) or hunger (foxes)" << endl << endl;
        
        // Borde superior
        cout << "+";
        for (int j = 0; j < grid_cols; j++) cout << "--";
        cout << "+" << endl;
        
        // Contenido de la grid
        for (int i = 0; i < grid_rows; i++) {
            cout << "|";
            for (int j = 0; j < grid_cols; j++) {
                auto creature = getCreatureAt(Position(i, j));
                if (creature == nullptr) {
                    cout << "  ";
                } else {
                    switch(creature->kind) {
                        case STONE:
                            cout << " *";
                            break;
                        case BUNNY: {
                            int age = generation - creature->generation_born;
                            if (age > 9) cout << "R+";
                            else cout << "R" << age;
                            break;
                        }
                        case PREDATOR: {
                            int hunger = generation - creature->last_meal_time;
                            if (hunger > 9) cout << "F+";
                            else cout << "F" << hunger;
                            break;
                        }
                        default:
                            cout << "  ";
                    }
                }
            }
            cout << "|" << endl;
        }
        
        // Borde inferior
        cout << "+";
        for (int j = 0; j < grid_cols; j++) cout << "--";
        cout << "+" << endl;
        
        // Pausa para visualización (0.5 segundos)
        usleep(500000);
    }
    
    void runSimulation() {
        omp_set_num_threads(omp_get_max_threads());
        
        // Mostrar estado inicial
        displayWorld(0);
        
        for (int gen = 0; gen < total_generations; gen++) {
            simulateBunnyGeneration(gen);
            simulatePredatorGeneration(gen);
            displayWorld(gen + 1);
        }
    }
    
    void outputFinalConfiguration() {
        cout << "\n=== FINAL CONFIGURATION ===" << endl;
        cout << bunny_reproduction_period << " " << predator_reproduction_period << " " 
             << predator_starvation_limit << " 0 " << grid_rows << " " << grid_cols << " " 
             << (stones.size() + bunnies.size() + predators.size()) << endl;
             
        for (int row = 0; row < grid_rows; row++) {
            for (int col = 0; col < grid_cols; col++) {
                auto creature = getCreatureAt(Position(row, col));
                if (creature != nullptr) {
                    switch(creature->kind) {
                        case STONE:
                            cout << "ROCK " << row << " " << col << endl;
                            break;
                        case BUNNY:
                            cout << "RABBIT " << row << " " << col << endl;
                            break;
                        case PREDATOR:
                            cout << "FOX " << row << " " << col << endl;
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
    
    void cleanup() {
        #pragma omp parallel sections
        {
            #pragma section
            { world_grid.clear(); }
            #pragma section
            { stones.clear(); }
            #pragma section
            { bunnies.clear(); }
            #pragma section
            { predators.clear(); }
        }
    }
};

int main() {
    EcosystemSimulator simulator;
    int br, pr, ps, gen, rows, cols, pop;
    cin >> br >> pr >> ps >> gen >> rows >> cols >> pop;
    
    simulator.initialize(br, pr, ps, gen, rows, cols, pop);
    
    for (int i = 0; i < pop; i++) {
        string creature_type;
        int row, col;
        cin >> creature_type >> row >> col;
        
        if (creature_type == "ROCK") {
            simulator.addCreature(STONE, row, col);
        } else if (creature_type == "RABBIT") {
            simulator.addCreature(BUNNY, row, col);
        } else if (creature_type == "FOX") {
            simulator.addCreature(PREDATOR, row, col);
        }
    }
    
    simulator.runSimulation();
    simulator.outputFinalConfiguration();
    simulator.cleanup();
    
    return 0;
}