import os
import sys 
sys.path.append(".")

# from models.mlp import MLP
from utils.weight_matching import PermutationSpec, permutation_spec_from_axes_to_perm, mlp_permutation_spec, weight_matching, apply_permutation
from utils.plot import plot_interp_acc
import argparse
import torch
from torchvision import datasets, transforms
from utils.utils import flatten_params, lerp
# from utils.training import test
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from net_pbc import *
from systems_pbc import *
from pinn_utils import *
# from visualize import *
# from PyHessian.pyhessian import hessian_pinn
import functions as func
from visualize_pinn import *


################################################################################
################################################################################

# u_pred = model.predict(X_star)

# error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
# error_u_abs = np.mean(np.abs(u_star - u_pred))
# error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)

# print('Error u rel: %e' % (error_u_relative))
# print('Error u abs: %e' % (error_u_abs))
# print('Error u linf: %e' % (error_u_linf))

def compute_loss_old(model, X):
    
    x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
    
    if torch.is_grad_enabled():
        model.optimizer.zero_grad()
    u_pred = model.dnn(torch.cat([x, t], dim=1))
    u_pred_lb = model.net_u(model.x_bc_lb, model.t_bc_lb)
    u_pred_ub = model.net_u(model.x_bc_ub, model.t_bc_ub)
    if model.nu != 0:
        u_pred_lb_x, u_pred_ub_x = model.net_b_derivatives(u_pred_lb, u_pred_ub, model.x_bc_lb, model.x_bc_ub)
    f_pred = model.net_f(model.x_f, model.t_f)
    
    if model.loss_style == 'mean':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.mean(f_pred ** 2)
    elif model.loss_style == 'sum':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.sum(f_pred ** 2)

    loss = loss_u + loss_b + model.L*loss_f
    # if j < len(model_coords):
    #     print(f"Loss value at {j}th model: {loss.detach().cpu().numpy()}")
    
    return float(loss.detach().cpu().numpy())


def compute_loss(model, X, data_loss=True, physics_loss=True, boundary_loss=True, L_eval=1.0):
    
    x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
    
    if torch.is_grad_enabled():
        model.optimizer.zero_grad()
    u_pred = model.dnn(torch.cat([x, t], dim=1))
    u_pred_lb = model.net_u(model.x_bc_lb, model.t_bc_lb)
    u_pred_ub = model.net_u(model.x_bc_ub, model.t_bc_ub)
    if model.nu != 0:
        u_pred_lb_x, u_pred_ub_x = model.net_b_derivatives(u_pred_lb, u_pred_ub, model.x_bc_lb, model.x_bc_ub)
    f_pred = model.net_f(model.x_f, model.t_f)
    
    if model.loss_style == 'mean':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.mean(f_pred ** 2)
    elif model.loss_style == 'sum':
        loss_u = torch.mean((t - u_pred) ** 2)
        loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
        if model.nu != 0:
            loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
        loss_f = torch.sum(f_pred ** 2)

    loss = loss_u * 0.0 # initialize zero tensor
    if data_loss:
        loss = loss_u 
    if boundary_loss:
        loss += loss_b
    if physics_loss:
        loss += L_eval * model.L*loss_f
    # if j < len(model_coords):
    #     print(f"Loss value at {j}th model: {loss.detach().cpu().numpy()}")
    
    return float(loss.detach().cpu().numpy())


def test(model, device, test_loader, softmax=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, acc))
    return test_loss, acc


################################################################################
################################################################################

# DNN(
#   (layers): Sequential(
#     (layer_0): Linear(in_features=2, out_features=50, bias=True)
#     (activation_0): Tanh()
#     (layer_1): Linear(in_features=50, out_features=50, bias=True)
#     (activation_1): Tanh()
#     (layer_2): Linear(in_features=50, out_features=50, bias=True)
#     (activation_2): Tanh()
#     (layer_3): Linear(in_features=50, out_features=50, bias=True)
#     (activation_3): Tanh()
#     (layer_4): Linear(in_features=50, out_features=1, bias=True)
#   )
# )

# dict_keys([
#    'layers.layer_0.weight', 'layers.layer_0.bias', 
#    'layers.layer_1.weight', 'layers.layer_1.bias',
#    'layers.layer_2.weight', 'layers.layer_2.bias', 
#   'layers.layer_3.weight', 'layers.layer_3.bias', 
#   'layers.layer_4.weight', 'layers.layer_4.bias'
# ])

def pinn_dnn_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
  """We assume that one permutation cannot appear in two axes of the same weight array."""
  assert num_hidden_layers >= 1
  return permutation_spec_from_axes_to_perm({
      "layers.layer_0.weight": ("P_0", None),
      **{f"layers.layer_{i}.weight": ( f"P_{i}", f"P_{i-1}")
         for i in range(1, num_hidden_layers)},
      **{f"layers.layer_{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      f"layers.layer_{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
      f"layers.layer_{num_hidden_layers}.bias": (None, ),
  })


################################################################################
################################################################################

def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_a", type=str, required=True)
    # parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--model_a_seed", type=str, required=True)
    parser.add_argument("--model_b_seed", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    ### PINN arguments
    parser.add_argument('--system', type=str, default='convection', help='System to study.')
    # parser.add_argument('--seed', type=int, default=0, help='Random initialization (for directions not models).')
    parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
    parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')
    parser.add_argument('--L_eval', type=float, default=1.0, help='Multiplier on loss f.')

    parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
    parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
    parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
    parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
    parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
    parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

    parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
    parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
    parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
    parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')
    
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--data_loss', type=int, default=1)
    parser.add_argument('--physics_loss', type=int, default=1)
    parser.add_argument('--boundary_loss', type=int, default=1)

    args,_ = parser.parse_known_args()


    ###############################################################################
    # Process the arguments
    ###############################################################################

    FLAG = False
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device('cuda')
        FLAG = True
    else:
        device = torch.device('cpu')

    nu = args.nu
    beta = args.beta
    rho = args.rho

    if args.system == 'diffusion': # just diffusion
        beta = 0.0
        rho = 0.0
    elif args.system == 'convection':
        nu = 0.0
        rho = 0.0
    elif args.system == 'rd': # reaction-diffusion
        beta = 0.0
    elif args.system == 'reaction':
        nu = 0.0
        beta = 0.0

    print('nu', nu, 'beta', beta, 'rho', rho)

    # parse the layers list here
    orig_layers = args.layers
    layers = [int(item) for item in args.layers.split(',')]

    ###############################################################################
    # Process the data
    ###############################################################################

    x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

    if 'convection' in args.system or 'diffusion' in args.system:
        u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'rd' in args.system:
        u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'reaction' in args.system:
        u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
    uu2 = Exact[:,0:1] # u(-end, t)

    # generate the other BC, now at x=2pi
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train = uu1 # just the initial condition
    X_u_train = xx1 # (x,t) for initial condition

    layers.insert(0, X_u_train.shape[-1])





    ###############################################################################
    # Load the models
    ###############################################################################

    # load models
    # model_a = PINN()
    # model_b = PINN()
    # checkpoint = torch.load(args.model_a)
    # model_a.dnn.load_state_dict(checkpoint)   
    # checkpoint_b = torch.load(args.model_b)
    # model_b.dnn.load_state_dict(checkpoint_b)

    # ../characterizing-pinns-failure-modes/pbc_examples/checkpoints/PINN_convection_beta_1.0_lr_1.0_seed_001.pt
    # /Users/calebgeniesse/vis4sciml/loss-landscape-profiles/generate_loss_cubes/saved_models/PINN_checkpoints

    def get_pinn_model(system,u0_str,nu,beta,rho,N_f,layers,L,source,seed,lr):
        base_path = f"/Users/calebgeniesse/vis4sciml/loss-landscape-profiles/generate_loss_cubes"
        model_path = f"{base_path}/saved_models/PINN_checkpoints/PINN_{system}/lr_{lr}/beta_{beta}/"
        model_name = f"pretrained_{system}_u0{u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_source{source}_seed{seed}.pt"
        model = torch.load(model_path+model_name, map_location=device)
        return model

    model_a = get_pinn_model(args.system,args.u0_str,nu,beta,rho,args.N_f,args.layers,args.L,args.source,args.model_a_seed,args.lr)
    model_b = get_pinn_model(args.system,args.u0_str,nu,beta,rho,args.N_f,args.layers,args.L,args.source,args.model_b_seed,args.lr)
    
    # model_a.dnn.eval()
    # model_b.dnn.eval()

    print(dict(model_a.dnn.named_parameters()).keys())
    print(model_a.dnn)





    ###############################################################################
    # Weight Matching
    ###############################################################################

    permutation_spec = pinn_dnn_permutation_spec(4)
    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a.dnn), flatten_params(model_b.dnn))
              

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b.dnn))

    


    ###############################################################################
    # Save new model
    ###############################################################################

    # save updated model
    model_b.dnn.load_state_dict(updated_params)


    if args.visualize:

        path = f"heatmap_results/{args.system}"
        if not os.path.exists(path):
            os.makedirs(path)

        u_pred = model_b.predict(X_star)
        u_pred = u_pred.reshape(len(t), len(x))
        exact_u(Exact, x, t, nu, beta, rho, orig_layers, args.N_f, args.L, args.source, args.u0_str, args.system, path=path)
        u_diff(Exact, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
        u_predict(u_vals, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
        plt.close('all')

    if args.save_model: # whether or not to save the model
        path = f"saved_models/lr_{args.lr}/beta_{beta}/"
        if not os.path.exists(path):
            os.makedirs(path)
        save_as = path + f"pretrained_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.model_b_seed}_rebasin_seed{args.model_a_seed}.pt"
        torch.save(model_b, save_as)
        print(f"[+] {save_as}")




    ###############################################################################
    # Evaluation
    ###############################################################################

    # loss landscape stuff
    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []


    # naive
    # model_b.dnn.load_state_dict(checkpoint_b)
    model_b = get_pinn_model(args.system,args.u0_str,nu,beta,rho,args.N_f,args.layers,args.L,args.source,args.model_b_seed,args.lr)
    model_a_dict = copy.deepcopy(model_a.dnn.state_dict())
    model_b_dict = copy.deepcopy(model_b.dnn.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.dnn.load_state_dict(naive_p)

        # test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
        # acc = compute_loss(model_b, X, data_loss=args.data_loss, physics_loss=args.physics_loss, boundary_loss=args.boundary_loss, L_eval=args.L_eval)
        u_pred = model_b.predict(X_star)
        acc = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        test_acc_interp_naive.append(acc)

        # train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
        # acc = compute_loss(model_b, X_test)
        # acc = compute_loss(model_b, X, data_loss=args.data_loss, physics_loss=args.physics_loss, boundary_loss=args.boundary_loss, L_eval=args.L_eval)
        train_acc_interp_naive.append(acc)


        if args.visualize:

            path = f"heatmap_results/{args.system}_linear_lam_{lam}"
            if not os.path.exists(path):
                os.makedirs(path)

            u_pred = model_b.predict(X_star)
            u_pred = u_pred.reshape(len(t), len(x))
            exact_u(Exact, x, t, nu, beta, rho, orig_layers, args.N_f, args.L, args.source, args.u0_str, args.system, path=path)
            u_diff(Exact, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
            u_predict(u_vals, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
            plt.close('all')


    # smart
    model_b.dnn.load_state_dict(updated_params)
    # model_b.cuda()
    # model_a.cuda()
    model_a_dict = copy.deepcopy(model_a.dnn.state_dict())
    model_b_dict = copy.deepcopy(model_b.dnn.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.dnn.load_state_dict(naive_p)

        # test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
        # acc = compute_loss(model_b, X, data_loss=args.data_loss, physics_loss=args.physics_loss, boundary_loss=args.boundary_loss, L_eval=args.L_eval)
        u_pred = model_b.predict(X_star)
        acc = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        test_acc_interp_clever.append(acc)

        # train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
        # acc = compute_loss(model_b.cuda(), X_test)
        # acc = compute_loss(model_b, X, data_loss=args.data_loss, physics_loss=args.physics_loss, boundary_loss=args.boundary_loss, L_eval=args.L_eval)
        train_acc_interp_clever.append(acc)


        if args.visualize:

            path = f"heatmap_results/{args.system}_rebasin_lam_{lam}"
            if not os.path.exists(path):
                os.makedirs(path)

            u_pred = model_b.predict(X_star)
            u_pred = u_pred.reshape(len(t), len(x))
            exact_u(Exact, x, t, nu, beta, rho, orig_layers, args.N_f, args.L, args.source, args.u0_str, args.system, path=path)
            u_diff(Exact, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
            u_predict(u_vals, u_pred, x, t, nu, beta, rho, args.model_b_seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path)
            plt.close('all')

    # final plotting
    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever)
    
    # TODO: format save_as 
    save_as = f"PINN_weight_matching_interp_accuracy_epoch.png"
    plt.savefig(save_as, dpi=300)
    print(f"[+] {save_as}")


if __name__ == "__main__":
  main()